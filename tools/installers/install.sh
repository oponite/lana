#!/bin/sh
# Release rendering pins the script and downloads to the same version.
main() (
    set -eu
    umask 022
    version='@LANA_VERSION@'
    base="https://github.com/oponite/lana/releases/download/v$version"
    fail() { printf 'Lana installation failed: %s\n' "$*" >&2; exit 1; }
    system=$(uname -s)
    machine=$(uname -m)
    case "$system:$machine" in
        Darwin:arm64|Darwin:x86_64) platform=macos-universal; checksum=shasum ;;
        Linux:x86_64) platform=linux-x64; checksum=sha256sum ;;
        Linux:aarch64|Linux:arm64) platform=linux-arm64; checksum=sha256sum ;;
        *) fail "unsupported platform: $system $machine" ;;
    esac
    for tool in curl "$checksum" tar awk cmp mktemp; do
        command -v "$tool" >/dev/null 2>&1 || fail "missing system tool: $tool"
    done
    : "${HOME:?HOME must be set}"
    parent="$HOME/.local/share"
    prefix="$parent/lana"
    previous="$parent/lana.previous"
    lock="$parent/.lana-install.lock"
    mkdir -p "$parent"
    mkdir "$lock" 2>/dev/null || fail "installer lock exists: $lock; inspect it before retrying"
    work=''; promoted=no; moved=no; committed=no
    cleanup() {
        status=$?
        trap - EXIT HUP INT TERM
        set +e
        recovered=yes
        if [ "$committed" != yes ]; then
            for number in 1 2; do
                if [ -f "$work/profile.$number.changed" ]; then
                    profile=$(cat "$work/profile.$number.path")
                    if [ -f "$work/profile.$number.original" ]; then
                        cp -p "$work/profile.$number.original" "$profile" || recovered=no
                    else
                        rm -f "$profile" || recovered=no
                    fi
                fi
            done
            if [ "$promoted" = yes ]; then rm -rf "$prefix" || recovered=no; fi
            if [ "$moved" = yes ]; then mv "$previous" "$prefix" || recovered=no; fi
        fi
        if [ "$recovered" = yes ]; then
            [ -z "$work" ] || rm -rf "$work"
            rmdir "$lock"
        else
            printf 'Recovery required: retain %s and %s; restore %s before retrying.\n' "$work" "$lock" "$previous" >&2
        fi
        [ -z "${replacement:-}" ] || rm -f "$replacement"
        exit "$status"
    }
    trap cleanup EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
    work=$(mktemp -d "$parent/.lana-install.XXXXXX")
    [ ! -e "$previous" ] && [ ! -L "$previous" ] || fail "previous installation needs inspection: $previous"
    [ ! -L "$prefix" ] || fail "refusing a symlink installation: $prefix"
    if [ -e "$prefix" ]; then
        [ -f "$prefix/.lana-install" ] && [ "$(cat "$prefix/.lana-install")" = lana-installer-v1 ] ||
            fail "refusing to replace an unrelated directory: $prefix"
    fi
    asset="lana-$version-$platform.tar.gz"
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL --retry 2 --connect-timeout 15 --max-time 180 "$base/$asset" -o "$work/$asset"
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL --retry 2 --connect-timeout 15 --max-time 60 "$base/SHA256SUMS" -o "$work/SHA256SUMS"
    digest=$(awk -v name="$asset" '$2 == name { count++; digest=$1 } END {
        if (count != 1 || length(digest) != 64 || digest ~ /[^a-fA-F0-9]/) exit 1; print tolower(digest)
    }' "$work/SHA256SUMS") || fail 'missing or invalid archive checksum'
    if [ "$checksum" = shasum ]; then
        actual=$(shasum -a 256 "$work/$asset" | awk '{print $1}')
    else
        actual=$(sha256sum "$work/$asset" | awk '{print $1}')
    fi
    [ "$actual" = "$digest" ] || fail 'archive checksum mismatch'
    tar -tzf "$work/$asset" > "$work/entries"
    awk ' !/^[a-zA-Z0-9_.\/-]+$/ || /^\// || /(^|\/)\.\.(\/|$)/ ||
        !/^(bin\/?|share\/?|bin\/.*|share\/.*|version[.]txt)$/ { exit 1 }
    ' "$work/entries" || fail 'unsafe archive path'
    tar -tvzf "$work/$asset" > "$work/types"
    awk 'substr($0,1,1) != "-" && substr($0,1,1) != "d" { exit 1 }' "$work/types" || fail 'archive contains links or special files'
    bundle="$work/bundle"
    mkdir "$bundle"
    tar --no-same-owner -xzf "$work/$asset" -C "$bundle"
    for file in bin/lana bin/lana-compiler.labc share/lana/stdlib/core.lana share/doc/lana/LICENSE version.txt; do
        [ -f "$bundle/$file" ] || fail "missing archive asset: $file"
    done
    [ -x "$bundle/bin/lana" ] || fail 'executable permission missing'
    [ "$(cat "$bundle/version.txt")" = "$version" ] || fail 'archive version mismatch'
    printf '%s\n' 'import "std/core" as core; assert(resolve(core.distribution([[7, 1]])) == 7, "installed stdlib");' > "$work/check.lana"
    (
        cd "$work"
        unset LANA_COMPILER_LABC LANA_STDLIB_DIR
        report=$("$bundle/bin/lana" version)
        case "$report" in "Lana $version (LABC v2,"*) ;; *) fail 'executable version mismatch' ;; esac
        "$bundle/bin/lana" run "$work/check.lana"
    )
    printf '%s\n' lana-installer-v1 > "$bundle/.lana-install"
    cat > "$work/path-block" <<'BLOCK'
# >>> Lana PATH >>>
case ":$PATH:" in
    *":$HOME/.local/share/lana/bin:"*) ;;
    *) export PATH="$HOME/.local/share/lana/bin:$PATH" ;;
esac
# <<< Lana PATH <<<
BLOCK
    configure_profile() {
        profile=$1; number=$2; depth=0
        while [ -L "$profile" ]; do
            depth=$((depth + 1))
            [ "$depth" -lt 40 ] || fail 'shell profile symlink loop'
            target=$(readlink "$profile")
            case "$target" in /*) profile=$target ;; *) profile="$(dirname "$profile")/$target" ;; esac
        done
        mkdir -p "$(dirname "$profile")"
        printf '%s\n' "$profile" > "$work/profile.$number.path"
        if [ -e "$profile" ]; then
            [ -f "$profile" ] || fail "shell profile is not a regular file: $profile"
            cp -p "$profile" "$work/profile.$number.original"
            if grep -Fq '# >>> Lana PATH >>>' "$profile"; then
                [ "$(grep -Fc '# >>> Lana PATH >>>' "$profile")" = 1 ] &&
                    [ "$(grep -Fc '# <<< Lana PATH <<<' "$profile")" = 1 ] || fail "invalid Lana PATH block: $profile"
                awk '/^# >>> Lana PATH >>>$/ { copy=1 } copy { print } /^# <<< Lana PATH <<</ { copy=0 }' "$profile" > "$work/existing-block"
                cmp -s "$work/path-block" "$work/existing-block" || fail "modified Lana PATH block: $profile"
                return
            fi
            cp -p "$profile" "$work/profile.$number.new"
        else
            : > "$work/profile.$number.new"
        fi
        printf '\n' >> "$work/profile.$number.new"
        cat "$work/path-block" >> "$work/profile.$number.new"
        # Stage next to the profile so replacing it stays on the same filesystem.
        replacement=$(mktemp "$(dirname "$profile")/.lana-profile.XXXXXX")
        cp -p "$work/profile.$number.new" "$replacement"
        : > "$work/profile.$number.changed"
        mv -f "$replacement" "$profile"
        replacement=''
    }
    if [ -e "$prefix" ]; then mv "$prefix" "$previous"; moved=yes; fi
    mv "$bundle" "$prefix"; promoted=yes
    case "${SHELL:-/bin/zsh}" in
        */zsh) configure_profile "${ZDOTDIR:-$HOME}/.zshrc" 1 ;;
        */bash)
            login_profile="$HOME/.profile"
            for candidate in "$HOME/.bash_profile" "$HOME/.bash_login"; do
                if [ -e "$candidate" ] || [ -L "$candidate" ]; then login_profile=$candidate; break; fi
            done
            configure_profile "$login_profile" 1
            configure_profile "$HOME/.bashrc" 2
            ;;
        *) printf 'Shell setup is manual for %s.\n' "${SHELL:-unknown}" ;;
    esac
    committed=yes
    if [ "$moved" = yes ]; then rm -rf "$previous" || printf 'Remove the old bundle after inspection: %s\n' "$previous" >&2; fi
    printf 'Installed Lana %s at %s\nOpen a new terminal, or run:\n  export PATH="$HOME/.local/share/lana/bin:$PATH"\n' "$version" "$prefix"
    existing=$(command -v lana || true)
    if [ -n "$existing" ] && [ "$existing" != "$prefix/bin/lana" ]; then
        printf 'Your current terminal resolves another lana: %s\n' "$existing"
    fi
)
main "$@"
