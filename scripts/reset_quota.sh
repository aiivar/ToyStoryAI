#!/usr/bin/env bash
# Сброс бесплатной квоты пользователя.
#
# Квота в этом проекте выводится из БД: пользователь без одной успешной
# записи типа `tale` (см. GenerationRepository.has_completed_story).
# Поэтому «сброс» = перевод успешных сказок в статус `failed`. Записи
# удалять не нужно: история прогонов остаётся видимой.
#
# Использование:
#   scripts/reset_quota.sh                 # сбросить квоту единственному пользователю
#   scripts/reset_quota.sh 206619495       # конкретному user_id
#   scripts/reset_quota.sh 206619495 bot.db
#
# Опции:
#   --list   только показать текущее состояние квоты, ничего не менять
#   --yes    не спрашивать подтверждение
#   --dry-run показать, что изменится, базу не трогать

set -euo pipefail

DB="${DB_PATH:-bot.db}"
DRY_RUN=0
ASSUME_YES=0
LIST_ONLY=0
USER_ID=""
POSITIONAL=()

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --list) LIST_ONLY=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    -*) echo "Неизвестный флаг: $arg" >&2; exit 2 ;;
    *) POSITIONAL+=("$arg") ;;
  esac
done

USER_ID="${POSITIONAL[0]:-}"
[ "${#POSITIONAL[@]}" -ge 2 ] && DB="${POSITIONAL[1]}"

if [ ! -f "$DB" ]; then
  echo "База не найдена: $DB" >&2
  echo "Передай путь вторым аргументом или задай DB_PATH." >&2
  exit 1
fi

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "Не найден sqlite3. Установи его: brew install sqlite" >&2
  exit 1
fi

show_state() {
  sqlite3 -header -column "$DB" "
    SELECT user_id,
           SUM(type = 'tale' AND status = 'success') AS completed_tales,
           SUM(type = 'tale' AND status = 'failed')   AS failed_tales,
           CASE WHEN SUM(type = 'tale' AND status = 'success') > 0
                THEN 'quota spent' ELSE 'quota available' END AS quota
    FROM generations
    GROUP BY user_id;
  "
}

# --- Список пользователей ---------------------------------------------------

if [ -z "$USER_ID" ]; then
  # Пользователь не указан: берём единственного, иначе просим выбрать.
  # macOS bash 3.2 не умеет mapfile — читаем через простой цикл.
  USERS=""
  while IFS= read -r line; do
    [ -n "$line" ] && USERS="$USERS$line
"
  done < <(sqlite3 "$DB" "SELECT DISTINCT user_id FROM generations ORDER BY user_id;")

  USER_COUNT="$(printf '%s' "$USERS" | grep -c . || true)"

  if [ "$USER_COUNT" -eq 0 ]; then
    echo "В базе нет ни одного пользователя — квота и так свободна."
    exit 0
  fi
  if [ "$USER_COUNT" -gt 1 ]; then
    echo "В базе несколько пользователей, укажи user_id:"
    show_state
    exit 1
  fi
  USER_ID="$(printf '%s' "$USERS" | tr -d '\n')"
  echo "Определён единственный пользователь: $USER_ID"
fi

if ! printf '%s' "$USER_ID" | grep -qE '^[0-9]+$'; then
  echo "user_id должен быть числом, получено: '$USER_ID'" >&2
  exit 2
fi

# --- Текущее состояние ------------------------------------------------------

echo
echo "Квота до сброса:"
show_state
echo

if [ "$LIST_ONLY" -eq 1 ]; then
  exit 0
fi

USED="$(sqlite3 "$DB" "
  SELECT COUNT(*) FROM generations
  WHERE user_id = '$USER_ID' AND type = 'tale' AND status = 'success';
")"

if [ "$USED" -eq 0 ]; then
  echo "У пользователя $USER_ID нет завершённых сказок — квота уже свободна."
  exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] Будут переведены в failed эти записи:"
  sqlite3 -header -column "$DB" "
    SELECT id, user_id, created_at, status
    FROM generations
    WHERE user_id = '$USER_ID' AND type = 'tale' AND status = 'success';
  "
  exit 0
fi

# --- Подтверждение ----------------------------------------------------------

if [ "$ASSUME_YES" -ne 1 ]; then
  echo "Будет переведено в failed записей: $USED"
  if [ ! -t 0 ]; then
    echo "Нет TTY для подтверждения — повтори с --yes." >&2
    exit 2
  fi
  read -r -p "Сбросить квоту пользователя $USER_ID? [y/N] " reply
  case "$reply" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Отменено."; exit 0 ;;
  esac
fi

# --- Бэкап и сброс ----------------------------------------------------------

BACKUP="${DB}.bak-$(date +%Y%m%d-%H%M%S)"
cp "$DB" "$BACKUP"
echo "Бэкап: $BACKUP"

sqlite3 "$DB" "
  UPDATE generations
  SET status = 'failed',
      error  = 'quota reset by admin'
  WHERE user_id = '$USER_ID' AND type = 'tale' AND status = 'success';
"

echo
echo "Квота после сброса:"
show_state
echo
echo "Готово. Перезапусти бота, если он держит состояние в памяти."
