# Сервис анализа видео кассы (API + веб-интерфейс)

FastAPI-приложение: принимает видео, ставит задачу в очередь и отдаёт результат в формате раздела 10 ТЗ. Анализ выполняет `pipeline.analyze.Analyzer`: один экземпляр на процесс, его делят `FORTE_WORKERS` рабочих потоков.

## Где что лежит (GPU-хост `forte-vlm`)

| Что | Где |
|---|---|
| Код | `~/forte/service/` (модуль `service.app`), анализ — `~/forte/pipeline/` |
| API-ключ | `~/forte/service/api_key.txt` (создаётся при первом запуске; переопределяется env `FORTE_API_KEY`) |
| Публичный адрес | постоянный: `https://216-81-248-201.sslip.io` и `http://216.81.248.201` (`forte-proxy.service`, `deploy/Caddyfile`); резервный quick tunnel: `~/forte/service/public_url.txt` (пишет `forte-tunnel.service` из лога cloudflared) |
| Задачи | `~/forte/jobs/<job_id>/`: `input.mov`, `status.json`, `result.json` |
| systemd | `forte-api.service` (uvicorn на 127.0.0.1:8000), `forte-proxy.service` (Caddy на портах 80 и 443), `forte-tunnel.service` (cloudflared quick tunnel) |
| Режим-заглушка | drop-in `/etc/systemd/system/forte-api.service.d/fake.conf` (`FORTE_FAKE=1`) |
| Логи | `journalctl -u forte-api -f`, `journalctl -u forte-proxy -f`, `journalctl -u forte-tunnel -f` |

## Запуск

```bash
# установка или обновление юнитов (идемпотентно): --fake ставит заглушку, --real убирает её
bash ~/forte/service/deploy/install.sh --fake
# вручную, без systemd
cd ~/forte && PYTHONPATH=$PWD ~/det-venv/bin/python -m uvicorn service.app:app --host 127.0.0.1 --port 8000
```

Переход с заглушки на настоящий анализатор:

```bash
sudo rm /etc/systemd/system/forte-api.service.d/fake.conf && sudo systemctl daemon-reload && sudo systemctl restart forte-api
curl -s 127.0.0.1:8000/api/v1/health   # "analyzer":"real","analyzer_state":"ready"
```

Модели загружаются в фоне после старта. Пока идёт загрузка, `analyzer_state` равен `loading`, а принятые задачи ждут в очереди. Если `Analyzer()` упал при загрузке, `analyzer_state` равен `error`, и каждая задача завершается со статусом `failed` и текстом причины.

Постоянный адрес отдаёт Caddy (`forte-proxy.service`): сертификат Let's Encrypt для `216-81-248-201.sslip.io` хранится в `/var/lib/forte-proxy` и переживает перезапуски, тело запроса — до 210 МБ, ufw открывает порты 80 и 443. Перезапуск `forte-api` адрес не меняет. Проверка: `curl -s https://216-81-248-201.sslip.io/api/v1/health`.

Резервный адрес quick tunnel меняется при каждом перезапуске `forte-tunnel.service`; текущий: `cat ~/forte/service/public_url.txt`. Cloudflare принимает тело запроса до 100 МБ (видео ТЗ — около 29 МБ) и не даёт гарантий доступности.

## Переменные окружения

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `FORTE_API_KEY` | — | ключ API; если не задан, берётся из `FORTE_API_KEY_FILE` |
| `FORTE_API_KEY_FILE` | `~/forte/service/api_key.txt` | файл ключа; создаётся со случайным ключом, если его нет |
| `FORTE_WORKERS` | `2` | число рабочих потоков (одновременных анализов) |
| `FORTE_FAKE` | — | `1` — заглушка `service/fake.py` вместо `pipeline.analyze.Analyzer` |
| `FORTE_FAKE_SEC` | `4` | длительность работы заглушки, с |
| `FORTE_DEVICE` | `cuda:0` | устройство для `Analyzer` |
| `FORTE_JOBS_DIR` | `~/forte/jobs` | каталог задач |
| `FORTE_MAX_UPLOAD_MB` | `200` | предельный размер загрузки |

В юните `forte-api.service` также заданы `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `YOLO_OFFLINE=1`, `YOLO_AUTOINSTALL=0` и `PYTHONPATH=/home/shadeform/forte`.

## API

| Метод | Путь | Ключ | Ответ |
|---|---|---|---|
| POST | `/api/v1/analyze` (multipart, поле `file`, .mov/.mp4) | да | `{"job_id": "...", "status": "processing"}` |
| GET | `/api/v1/jobs/{job_id}` | да | `status`: `processing` (+`progress` 0–1, `stage`, `queue_position`) · `done` (+`verdict`, `violations`, `meta`) · `failed` (+`error`) |
| GET | `/api/v1/health` | нет | `{"ok": true, "queue": n, "workers": n, ...}` |
| GET | `/api/v1/types` (то же: `/ui/types`) | нет | справочник типов нарушений из `pipeline/norms.json`: `type_id`, `name`, `summary`, `strength`, `rule`, `extra_rules`, а также `sources` |

Ошибки: 401 — нет ключа или ключ неверный; 404 — неизвестный `job_id`; 413 — файл больше лимита; 400 — нет поля `file`, файл не .mov/.mp4, пустой файл, ffprobe не находит видеопоток или ffmpeg не декодирует кадр в начале или в середине ролика (до 5 с на кадр). Последняя проверка отсеивает файлы с целым заголовком и испорченными данными на этапе загрузки, а не после постановки задачи.

Каждое нарушение содержит поля ТЗ (`type`, `timecode_sec`, `description`, `evidence_frames[{timestamp_sec, image_base64}]`) и дополнительные поля `type_id`, `rule {document, short, clause, url, quote, strength}`, `confidence`. В `meta` лежат `duration_sec`, `processing_sec`, `stage_sec`, `models`, `frames_analyzed`.

Проверка загрузки на работающем экземпляре: `bash service/deploy/test_upload_check.sh http://127.0.0.1:8000 "$KEY"`. Скрипт загружает все ролики из `~/forte/videos` (ставит их в очередь!) и испорченную копию X-02.

```bash
URL=https://216-81-248-201.sslip.io; KEY=$(cat ~/forte/service/api_key.txt)
curl -X POST "$URL/api/v1/analyze" -H "X-API-Key: $KEY" -F "file=@video_017.mov"
# {"job_id":"99c7837d0400","status":"processing"}
curl -H "X-API-Key: $KEY" "$URL/api/v1/jobs/99c7837d0400"
```

## Веб-интерфейс

`GET /` открывает страницу без ключа (FAQ 27). Она обращается к тем же обработчикам по маршрутам без ключа: `POST /ui/analyze`, `GET /ui/jobs/{id}`, `GET /ui/jobs/{id}/video` (исходное видео для плеера), `GET /ui/types`. Статика лежит в `service/static/`: HTML, JS и CSS без сборки; сервер отдаёт её с диска, поэтому после правки статики перезапуск не нужен. Ссылка `/#job=<id>` открывает результат задачи.

Что есть на странице:
- загрузка перетаскиванием или через выбор файла; отказ (формат, размер, нечитаемое видео) показывается с причиной от сервера;
- ход обработки: этап, процент, место в очереди, прошедшее время;
- результат: баннер вердикта, имя видео, длительность, время обработки, модели (`meta.models`), примечание о проверке человеком;
- плеер и шкала событий под ним: маркер на каждое нарушение (цвет по типу) и текущая позиция; клик по маркеру или по таймкоду в карточке перематывает видео;
- карточки нарушений: тип и `type_id`, таймкод, уверенность, описание, норма (краткое название, пункт, ссылка на adilet, дословный текст в раскрывающемся блоке, связь «прямая/косвенная»), кадры с просмотром во весь экран и листанием (стрелки, Esc);
- окно «Типы нарушений» со справочником из `/ui/types`;
- «Недавние проверки»: последние 20 задач этого браузера (localStorage).
