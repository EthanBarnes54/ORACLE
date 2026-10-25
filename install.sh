#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  ORACLE - Installer (Linux/macOS)
#  Online Readout & Acquisition for Collinear Laser spectroscopy Experiments
#
#  Usage:
#    ./install.sh            Interactive menu
#    ./install.sh full       Full install (uv + dependencies + shortcut)
#    ./install.sh rack       Full install plus the instrument libraries
#    ./install.sh uv         Install uv only
#    ./install.sh shortcut   Create desktop shortcut only
# ─────────────────────────────────────────────────────────────

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

echo "==========================================================="
echo "  ORACLE - Online Readout & Acquisition for CLS Experiments"
echo "==========================================================="
echo

install_uv() {
    if command -v uv &>/dev/null; then
        echo "uv already installed: $(uv --version)"
        return 0
    fi
    echo "uv was not found on PATH."
    echo
    read -rp "Install uv now via the official Astral installer? [Y/n]: " ans
    case "$ans" in
        n|N)
            echo
            echo "Cannot continue without uv. Install manually from"
            echo "  https://docs.astral.sh/uv/getting-started/installation/"
            return 1
            ;;
    esac
    echo
    echo "Installing uv..."
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        echo
        echo "ERROR: uv installation failed."
        return 1
    fi
    # uv installs to ~/.local/bin; pick it up for this shell
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo
        echo "uv was installed but is not yet visible in this shell's PATH."
        echo "Open a NEW terminal and re-run ./install.sh."
        return 1
    fi
    return 0
}

sync_deps() {
    echo
    echo "-----------------------------------------------------------"
    echo "  Installing Python + dependencies via uv sync..."
    echo "-----------------------------------------------------------"
    echo
    # --managed-python: a venv on a system Python breaks when that Python is
    # upgraded underneath it
    if ! uv sync --managed-python "$@"; then
        echo
        echo "ERROR: uv sync failed. See output above."
        return 1
    fi
    echo
    echo "Dependencies installed into .venv"
    return 0
}

create_shortcut() {
    local GUI_PY="$APP_DIR/gui.py"
    local ICON="$APP_DIR/icons/oracle_512.png"
    local PYTHON="$APP_DIR/.venv/bin/python"
    if [ ! -x "$PYTHON" ]; then
        echo
        echo "ERROR: Project venv not found at .venv/"
        echo 'Run "./install.sh full" first to set up uv and dependencies.'
        return 1
    fi
    local TARGET="$HOME/Desktop"
    mkdir -p "$TARGET"
    local SC
    case "$(uname -s)" in
        Darwin)
            # iCloud keeps setting the hidden flag on files in a .venv under
            # ~/Documents, and Qt skips hidden plugins, so the app would not
            # start: keep the environment out of iCloud and clear its flags
            xattr -w 'com.apple.fileprovider.ignore#P' 1 "$APP_DIR/.venv" || return 1
            chflags -R nohidden "$APP_DIR/.venv"
            # A minimal app bundle in ~/Applications, linked from the Desktop,
            # so ORACLE runs with its own icon and no Terminal window.
            local APP="$HOME/Applications/ORACLE.app"
            local RES="$APP/Contents/Resources"
            local VER
            VER="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)"
            mkdir -p "$APP/Contents/MacOS" "$RES/oracle.iconset" || return 1
            for s in 16 32 128 256 512; do
                sips -z $s $s "$ICON" --out "$RES/oracle.iconset/icon_${s}x${s}.png" >/dev/null
                [ $s -lt 512 ] && sips -z $((s * 2)) $((s * 2)) "$ICON" \
                    --out "$RES/oracle.iconset/icon_${s}x${s}@2x.png" >/dev/null
            done
            iconutil -c icns "$RES/oracle.iconset" -o "$RES/oracle.icns" || return 1
            rm -f "$RES"/oracle.iconset/icon_*.png && rmdir "$RES/oracle.iconset"
            cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
    <key>CFBundleName</key><string>ORACLE</string>
    <key>CFBundleExecutable</key><string>ORACLE</string>
    <key>CFBundleIconFile</key><string>oracle</string>
    <key>CFBundleIdentifier</key><string>io.github.ethanbarnes54.oracle</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$VER</string>
    <key>NSHighResolutionCapable</key><true/>
</dict></plist>
EOF
            SC="$APP/Contents/MacOS/ORACLE"
            cat > "$SC" <<EOF
#!/usr/bin/env bash
cd "$APP_DIR" && exec "$PYTHON" "$GUI_PY" "\$@"
EOF
            chmod +x "$SC"
            SC="$TARGET/ORACLE.app"   # what the user double-clicks
            ln -sfn "$APP" "$SC"
            ;;
        *)
            SC="$TARGET/ORACLE.desktop"
            cat > "$SC" <<EOF
[Desktop Entry]
Type=Application
Name=ORACLE
Comment=Online Readout & Acquisition for Collinear Laser spectroscopy Experiments
Exec=$PYTHON $GUI_PY
Path=$APP_DIR
Icon=$ICON
Terminal=false
Categories=Science;Physics;
EOF
            chmod +x "$SC"
            ;;
    esac
    echo "Shortcut created: $SC"
    echo "Python:           $PYTHON"
    return 0
}

MODE="${1:-}"
DO_UV=0; DO_SYNC=0; DO_SC=0; SYNC_ARGS=()
case "$MODE" in
    full)     DO_UV=1; DO_SYNC=1; DO_SC=1 ;;
    rack)     DO_UV=1; DO_SYNC=1; DO_SC=1; SYNC_ARGS=(--group rack) ;;
    uv)       DO_UV=1 ;;
    shortcut) DO_SC=1 ;;
    "")
        echo "Choose an action:"
        echo "  [1] Full install (uv + dependencies + desktop shortcut)"
        echo "  [2] Full install for the beamline PC (adds the instrument libraries)"
        echo "  [3] Install uv only"
        echo "  [4] Create desktop shortcut only (assumes .venv exists)"
        echo "  [q] Quit"
        echo
        read -rp "Selection [1/2/3/4/q]: " CHOICE
        case "$CHOICE" in
            1) DO_UV=1; DO_SYNC=1; DO_SC=1 ;;
            2) DO_UV=1; DO_SYNC=1; DO_SC=1; SYNC_ARGS=(--group rack) ;;
            3) DO_UV=1 ;;
            4) DO_SC=1 ;;
            q|Q) exit 0 ;;
            *) echo "Invalid choice."; exit 1 ;;
        esac
        ;;
    *)
        echo "Unknown option '$MODE'. Use: full | rack | uv | shortcut"
        exit 1
        ;;
esac

[ "$DO_UV"   -eq 1 ] && { install_uv               || exit 1; }
[ "$DO_SYNC" -eq 1 ] && { sync_deps "${SYNC_ARGS[@]}" || exit 1; }
[ "$DO_SC"   -eq 1 ] && { create_shortcut          || exit 1; }

echo
echo "==========================================================="
echo "  Done. Launch from terminal:  uv run python gui.py"
echo "==========================================================="
