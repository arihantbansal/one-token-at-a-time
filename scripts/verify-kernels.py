#!/usr/bin/env python3
"""Repeat the guide's Linux-only Triton interpreter or sm_90 compiler checks."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys


def interpret(kernels, constants):
    import torch
    import triton

    project_toy, project, norm_gate, attend_slice = kernels
    torch.manual_seed(0)

    def check(name, got, want, tolerance):
        torch.testing.assert_close(got.double(), want.double(), rtol=0, atol=tolerance)
        error = (got.double() - want.double()).abs().max().item()
        print(f'PASS {name}: max error {error:.3g} (tolerance {tolerance:.3g})', flush=True)

    weights = torch.randn(3, 4)
    inputs = torch.randn(4)
    output = torch.empty(3)
    project_toy[(3,)](weights, inputs, output, 4, BLOCK_K=4)
    check('toy projection', output, weights.double() @ inputs.double(), 1e-5)

    block_n, block_k = constants['PROJ_BLOCK_N'], constants['PROJ_BLOCK_K']
    for rows, cols in [(constants['rows'], constants['cols']), (1000, 300)]:
        weights = torch.randn(rows, cols, dtype=torch.bfloat16)
        inputs = torch.randn(cols, dtype=torch.bfloat16)
        output = torch.empty(rows, dtype=torch.bfloat16)
        project[(triton.cdiv(rows, block_n),)](
            weights, inputs, output, rows, cols, BLOCK_N=block_n, BLOCK_K=block_k)
        reference = weights.double() @ inputs.double()
        check(f'tiled projection {rows}x{cols}', output, reference, 0.02 * reference.abs().max().item())
        edited = torch.empty_like(output)
        project[(triton.cdiv(rows, block_n // 2),)](
            weights, inputs, edited, rows, cols, BLOCK_N=block_n // 2, BLOCK_K=block_k)
        check(f'tile-height edit identical {rows}x{cols}', edited, output, 0)

    heads, width = constants['Q_HEADS'], constants['HEAD_DIM']
    values = torch.randn(heads, width, dtype=torch.bfloat16)
    gate = torch.randn_like(values)
    output = torch.empty_like(values)
    norm_gate[(heads,)](values, gate, output, 1e-5, D=width)
    reference = values.double() / torch.sqrt(values.double().square().mean(1, keepdim=True) + 1e-5)
    reference *= torch.sigmoid(gate.double())
    check('fused gated norm', output, reference, 0.02)

    for length, splits in [
            (1000, constants['ATTN_SPLITS']),
            (constants['CONTEXT_TOKENS'], constants['ATTN_SPLITS']), (5, 4), (3, 8)]:
        heads = 4
        query = torch.randn(heads, width, dtype=torch.bfloat16)
        keys = torch.randn(heads, length, width, dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        parts = torch.empty(heads, splits, width)
        lse = torch.empty(heads, splits)
        scale = 1 / math.sqrt(width)
        attend_slice[(heads, splits)](
            query, keys, values, parts, lse, length, triton.cdiv(length, splits), scale,
            SPLITS=splits, D=width, BLOCK=constants['ATTN_BLOCK'])
        scores = (keys.double() @ query.double()[..., None]).squeeze(-1) * scale
        reference = (torch.softmax(scores, -1)[..., None] * values.double()).sum(1)
        # Empty slices store NaN blends and -inf log totals; mask the blend before weighting it.
        weights = torch.exp(lse.double() - lse.double().max(1, keepdim=True).values)
        live_parts = torch.where(torch.isfinite(lse)[..., None], parts.double(), 0)
        combined = (weights[..., None] * live_parts).sum(1) / weights.sum(1, keepdim=True)
        check(f'split attention length={length} splits={splits}', combined, reference, 1e-4)


def compile_kernels(kernels, constants):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    project_toy, project, norm_gate, attend_slice = kernels
    cases = [
        ('toy', project_toy,
         {'w_ptr': '*fp32', 'x_ptr': '*fp32', 'y_ptr': '*fp32', 'K': 'i32', 'BLOCK_K': 'constexpr'},
         {'BLOCK_K': 4}),
        ('tiled', project,
         {'w_ptr': '*bf16', 'x_ptr': '*bf16', 'y_ptr': '*bf16', 'N': 'i32', 'K': 'i32',
          'BLOCK_N': 'constexpr', 'BLOCK_K': 'constexpr'},
         {'BLOCK_N': constants['PROJ_BLOCK_N'], 'BLOCK_K': constants['PROJ_BLOCK_K']}),
        ('fused', norm_gate,
         {'a_ptr': '*bf16', 'g_ptr': '*bf16', 'out_ptr': '*bf16', 'eps': 'fp32', 'D': 'constexpr'},
         {'D': constants['HEAD_DIM']}),
        ('slice', attend_slice,
         {'q_ptr': '*bf16', 'k_ptr': '*bf16', 'v_ptr': '*bf16', 'part_ptr': '*fp32', 'lse_ptr': '*fp32',
          'length': 'i32', 'slice_len': 'i32', 'scale': 'fp32', 'SPLITS': 'constexpr',
          'D': 'constexpr', 'BLOCK': 'constexpr'},
         {'SPLITS': constants['ATTN_SPLITS'], 'D': constants['HEAD_DIM'], 'BLOCK': constants['ATTN_BLOCK']}),
    ]
    for name, kernel, signature, constexprs in cases:
        result = triton.compile(
            ASTSource(fn=kernel, signature=signature, constexprs=constexprs),
            target=GPUTarget('cuda', 90, 32))
        if not result.asm.get('cubin'):
            raise RuntimeError(f'{name}: compiler produced no cubin; inspect the compilation stages')
        print(f'PASS compile {name} for sm_90: stages {sorted(result.asm)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('interpret', 'compile'))
    parser.add_argument('directory', nargs='?', type=Path, default=Path('.astro/kernel-check'))
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Triton checks require Linux; see the Linux environment commands in README.md')
    os.environ['TRITON_INTERPRET'] = '1' if args.mode == 'interpret' else '0'
    directory = args.directory.resolve()
    if not (directory / 'consts.json').is_file():
        parser.error('extracted kernels are missing; run bun scripts/extract-kernels.ts first')
    constants = json.loads((directory / 'consts.json').read_text())
    sys.path.insert(0, str(directory))
    kernels = [getattr(importlib.import_module(module), function) for module, function in (
        ('k_toy', 'project'), ('k_tiled', 'project'), ('k_fused', 'norm_gate'), ('k_slice', 'attend_slice'))]
    if args.mode == 'interpret':
        interpret(kernels, constants)
    else:
        compile_kernels(kernels, constants)


if __name__ == '__main__':
    main()
