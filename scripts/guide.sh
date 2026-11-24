#!/bin/sh
# Open or close a local copy of the guide in the browser.
# usage: scripts/guide.sh [open|close|status|logs]
set -eu
cd "$(dirname "$0")/.."

astro() { bunx astro "$@"; }
# The URL of the background server this project started, if it is running. Astro picks the
# next free port when 4321 is taken, so read it rather than assume it.
url() { astro dev status 2>/dev/null | grep -o 'http://[^ "]*' | head -n 1 || true; }

case "${1:-open}" in
open)
	[ -d node_modules ] || bun install
	[ -n "$(url)" ] || astro dev --background
	u="$(url)"
	if [ -z "$u" ]; then
		echo "The dev server didn't start. See: $0 logs" >&2
		exit 1
	fi
	echo "Running at $u"
	if command -v open >/dev/null 2>&1; then open "$u"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$u"; fi
	;;
close)
	if [ -n "$(url)" ]; then astro dev stop; else echo "Not running."; fi
	;;
status) astro dev status ;;
logs) astro dev logs --follow ;;
*)
	echo "usage: $0 [open|close|status|logs]" >&2
	exit 2
	;;
esac
