#!/usr/bin/env bash
# Connects adb to a phone on the same Wi-Fi, then can install an apk.
# Pair once while the phone shows a pairing code:
#   android/connect_phone.sh pair 192.168.1.20:37123 123456
# Later sessions:
#   android/connect_phone.sh
#   android/connect_phone.sh install
#   android/connect_phone.sh adb shell getprop ro.product.model
set -euo pipefail

ANDROID_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${ANDROID_DIR}/.." && pwd)"
DEFAULT_APK="${ROOT}/android/app/build/outputs/apk/release/app-release.apk"

usage() {
  cat <<'EOF'
Usage:
  android/connect_phone.sh
  android/connect_phone.sh connect [HOST:PORT]
  android/connect_phone.sh pair HOST:PORT CODE
  android/connect_phone.sh install [APK]
  android/connect_phone.sh adb [adb arguments]

Pair once from the phone's Wireless debugging dialog. Later commands find the
phone on the local network. install sends the app release apk unless you pass
another path. adb runs the SDK adb after connecting.
EOF
}

sdk_dir() {
  local props line
  props="${ANDROID_DIR}/local.properties"
  if [[ ! -f "${props}" ]]; then
    return 1
  fi
  line="$(grep -E '^sdk\.dir=' "${props}" | head -n 1 || true)"
  line="${line#sdk.dir=}"
  line="${line//\\:/:}"
  line="${line%$'\r'}"
  if [[ -z "${line}" ]]; then
    return 1
  fi
  printf '%s\n' "${line}"
}

find_adb() {
  local sdk
  if [[ -n "${ADB:-}" && -x "${ADB}" ]]; then
    printf '%s\n' "${ADB}"
    return
  fi
  if sdk="$(sdk_dir)" && [[ -x "${sdk}/platform-tools/adb" ]]; then
    printf '%s\n' "${sdk}/platform-tools/adb"
    return
  fi
  if command -v adb >/dev/null 2>&1; then
    command -v adb
    return
  fi
  echo "adb was not found. Set sdk.dir in android/local.properties." >&2
  exit 1
}

require_endpoint() {
  local endpoint="$1"
  if [[ ! "${endpoint}" =~ :[0-9]+$ ]]; then
    echo "Expected host:port, got ${endpoint}." >&2
    exit 1
  fi
}

# An attached emulator is not the phone this script installs to.
online_serials() {
  "${ADB_BIN}" devices | awk 'NR > 1 && $2 == "device" && $1 !~ /^emulator-/ { print $1 }'
}

tls_serials() {
  online_serials | awk '/_adb-tls-connect\._tcp$/ { print }'
}

wait_for_tls() {
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if [[ -n "$(tls_serials)" ]]; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

# adb connect HOST:PORT adds a second serial beside the mDNS TLS serial.
keep_tls_serial() {
  local serial
  if [[ -z "$(tls_serials)" ]]; then
    return 0
  fi
  while IFS= read -r serial; do
    [[ -n "${serial}" ]] || continue
    case "${serial}" in
      *_adb-tls-connect._tcp) ;;
      *:*)
        "${ADB_BIN}" disconnect "${serial}" >&2 || true
        ;;
    esac
  done <<EOF
$(online_serials)
EOF
}

print_wifi_help() {
  cat <<'EOF' >&2
No phone is connected.

On the phone, open Developer options and turn on Wireless debugging.
Tap Pair device with pairing code.
While that dialog is open, run:
  android/connect_phone.sh pair HOST:PORT CODE

HOST:PORT and the six-digit code are on the dialog.

After this Mac is paired, turn Wireless debugging on and run:
  android/connect_phone.sh

If the phone shows IP address & Port and discovery still finds nothing, run:
  android/connect_phone.sh connect HOST:PORT
EOF
}

discover_endpoints() {
  local attempt line
  for attempt in 1 2 3 4 5; do
    line="$("${ADB_BIN}" mdns services | awk -F '[[:space:]]+' '$2 == "_adb-tls-connect._tcp" { print $3 }')"
    if [[ -n "${line}" ]]; then
      printf '%s\n' "${line}"
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_online() {
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if [[ -n "$(online_serials)" ]]; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

connect_reported_success() {
  local output="$1"
  local line
  while IFS= read -r line; do
    case "${line}" in
      "connected to "* | "already connected to "*)
        return 0
        ;;
    esac
  done <<< "${output}"
  return 1
}

connect_endpoint() {
  local endpoint="$1"
  local output
  require_endpoint "${endpoint}"
  output="$("${ADB_BIN}" connect "${endpoint}" 2>&1 || true)"
  printf '%s\n' "${output}" >&2
  if connect_reported_success "${output}"; then
    return 0
  fi
  if [[ "${output}" == *authenticate* ]]; then
    echo "This Mac is not paired with that phone yet." >&2
    print_wifi_help
  fi
  exit 1
}

connect_phone() {
  local endpoint="${1:-}"
  local found
  if [[ -n "${endpoint}" ]]; then
    "${ADB_BIN}" start-server >&2
    wait_for_tls || true
    if [[ -z "$(tls_serials)" ]]; then
      connect_endpoint "${endpoint}"
      wait_for_tls || true
    fi
    keep_tls_serial
    wait_for_online || true
    "${ADB_BIN}" devices -l >&2
    if [[ -z "$(online_serials)" ]]; then
      echo "The phone did not come online." >&2
      exit 1
    fi
    return
  fi
  if [[ -n "$(online_serials)" ]]; then
    "${ADB_BIN}" devices -l >&2
    return
  fi
  if ! found="$(discover_endpoints)"; then
    print_wifi_help
    exit 1
  fi
  wait_for_online || true
  if [[ -z "$(online_serials)" ]]; then
    while IFS= read -r endpoint; do
      [[ -n "${endpoint}" ]] || continue
      connect_endpoint "${endpoint}"
    done <<< "${found}"
  fi
  wait_for_tls || true
  keep_tls_serial
  if ! wait_for_online; then
    print_wifi_help
    exit 1
  fi
  "${ADB_BIN}" devices -l >&2
}

pair_phone() {
  local endpoint="${1:-}"
  local code="${2:-}"
  if [[ -z "${endpoint}" || -z "${code}" ]]; then
    usage >&2
    exit 1
  fi
  require_endpoint "${endpoint}"
  if [[ ! "${code}" =~ ^[0-9]{6}$ ]]; then
    echo "The pairing code must be six digits." >&2
    exit 1
  fi
  "${ADB_BIN}" pair "${endpoint}" "${code}"
  echo "Paired. Looking for the phone's Wireless debugging port."
  if ! found="$(discover_endpoints)"; then
    echo "Discovery did not see the phone yet." >&2
    echo "Read IP address & Port on the Wireless debugging screen, then run:" >&2
    echo "  android/connect_phone.sh connect HOST:PORT" >&2
    exit 1
  fi
  wait_for_online || true
  if [[ -z "$(online_serials)" ]]; then
    while IFS= read -r endpoint; do
      [[ -n "${endpoint}" ]] || continue
      connect_endpoint "${endpoint}"
    done <<< "${found}"
  fi
  wait_for_tls || true
  keep_tls_serial
  wait_for_online || true
  "${ADB_BIN}" devices -l >&2
  if [[ -z "$(online_serials)" ]]; then
    echo "The phone did not come online." >&2
    exit 1
  fi
}

choose_serial() {
  local count
  if [[ -n "${ANDROID_SERIAL:-}" ]]; then
    if online_serials | grep -Fxq "${ANDROID_SERIAL}"; then
      printf '%s\n' "${ANDROID_SERIAL}"
      return
    fi
    echo "ANDROID_SERIAL=${ANDROID_SERIAL} is not an online phone." >&2
    exit 1
  fi
  count="$(online_serials | wc -l | tr -d ' ')"
  if [[ "${count}" == "1" ]]; then
    online_serials
    return
  fi
  if [[ "${count}" == "0" ]]; then
    echo "No phone is online." >&2
    exit 1
  fi
  echo "More than one phone is online. Set ANDROID_SERIAL and run the command again." >&2
  online_serials >&2
  exit 1
}

install_apk() {
  local apk="${1:-${DEFAULT_APK}}"
  local serial
  if [[ ! -f "${apk}" ]]; then
    echo "APK not found: ${apk}" >&2
    echo "Build the app with: (cd android && ./gradlew :app:assembleRelease)" >&2
    exit 1
  fi
  connect_phone
  serial="$(choose_serial)"
  "${ADB_BIN}" -s "${serial}" install -r "${apk}"
}

ADB_BIN="$(find_adb)"
command="${1:-connect}"

case "${command}" in
  connect)
    shift || true
    connect_phone "${1:-}"
    ;;
  pair)
    shift
    pair_phone "${1:-}" "${2:-}"
    ;;
  install)
    shift || true
    install_apk "${1:-}"
    ;;
  adb)
    shift
    connect_phone
    "${ADB_BIN}" "$@"
    ;;
  -h | --help | help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
