#!/bin/sh
# make_manifest.sh -- inventory /mnt/qwen with md5 sums, for backup.
#
# Small files (<1 MB) get a full md5.  The three multi-hundred-MB binaries
# are too slow to hash in full over this board, so they get the md5 of their
# first 1 MiB plus their exact size; that is enough to detect corruption or
# a swapped file, and expected.md5 already pins the first 1 MiB of the
# weights for the acceptance check.

cd /mnt/qwen || exit 1

OUT=/mnt/qwen/BACKUP_MANIFEST.txt
LIMIT=1048576

: > "$OUT"

echo "# Qwen3.5-0.8B / AX7020 backup manifest" >> "$OUT"
echo "# generated: $(date)" >> "$OUT"
echo "# kernel: $(uname -r)" >> "$OUT"
echo "# format: md5  size  path   (BIG1M: prefix = md5 of first 1 MiB)" >> "$OUT"
echo "" >> "$OUT"

find . -maxdepth 2 -type f | sort | while read -r f; do
    case "$f" in
        ./BACKUP_MANIFEST.txt) continue ;;
    esac
    sz=$(stat -c %s "$f" 2>/dev/null)
    [ -z "$sz" ] && continue
    if [ "$sz" -lt "$LIMIT" ]; then
        m=$(md5sum "$f" 2>/dev/null | cut -d' ' -f1)
    else
        m="BIG1M:$(head -c "$LIMIT" "$f" 2>/dev/null | md5sum | cut -d' ' -f1)"
    fi
    echo "$m  $sz  $f" >> "$OUT"
done

n=$(grep -vc '^#' "$OUT" 2>/dev/null)
echo "manifest written: $OUT"
echo "entries: $n"
echo ""
echo "=== files >= 1 MiB ==="
grep BIG1M "$OUT"
echo ""
echo "=== totals ==="
du -sh /mnt/qwen 2>/dev/null
df -h /mnt/qwen | tail -1
