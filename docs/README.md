# Контроль кассовой дисциплины по видео — ForteBank Datathon 26

Сервис принимает видеозапись кассы (.mov, камера сверху) и сообщает, видны ли на ней признаки возможных нарушений кассовой дисциплины. Для каждого найденного случая он выдаёт тип, таймкод, описание того, что видно на кадрах, 1–3 кадра‑доказательства и пункт нормативного акта РК. Все модели работают на сервере команды, облачные AI‑сервисы не используются (раздел 7 ТЗ). Результат — подсказка для проверки человеком, а не заключение.

## Документация

| Файл | Содержание |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | схема обработки, модели и почему они, устройство сервиса, локальность, время обработки |
| [TYPES.md](TYPES.md) | типы нарушений T1–T13 и GENERIC: что ищем, по каким сигналам, таймкод, норма, что проверено на открытых роликах |
| [NORMS.md](NORMS.md) | нормативные источники, принцип сопоставления «действие → тип → пункт», цитаты |
| [LIMITATIONS.md](LIMITATIONS.md) | ограничения, известные ошибки, чего сервис не утверждает |
| [API.md](API.md) | API для жюри: запросы, ответы, ошибки, опрос статуса |
| [../service/README.md](../service/README.md) | эксплуатация сервиса на GPU‑хосте: юниты, переменные окружения, логи |

## Адреса и ключ

- **Веб‑интерфейс:** `https://216-81-248-201.sslip.io/`, ключ не нужен (FAQ 27).
- **API:** `https://216-81-248-201.sslip.io/api/v1/...`, ключ в заголовке `X-API-Key` (раздел 10 ТЗ).
- **Постоянный адрес** (18.10.2026, с 09:40 UTC+5): `https://216-81-248-201.sslip.io`, сертификат Let's Encrypt. Запасной без TLS — `http://216.81.248.201`. Оба адреса обслуживает обратный прокси Caddy на самом GPU‑хосте (`forte-proxy.service`, порты 80 и 443) и не меняются при перезапусках. Имя `216-81-248-201.sslip.io` — публичный DNS sslip.io, который сопоставляет имя с IP‑адресом хоста.
- **Резервный адрес:** Cloudflare quick tunnel (`forte-tunnel.service`); адрес меняется при каждом перезапуске туннеля, текущий лежит на хосте в `~/forte/service/public_url.txt`.
- **API‑ключ** в документацию не вписан. Он лежит на GPU‑хосте в `~/forte/service/api_key.txt` (создаётся при первом запуске сервиса, можно задать переменной `FORTE_API_KEY`). Команда передаёт ключ жюри вместе с адресом.

Проверка, что сервис жив и работает настоящий анализатор:

```bash
curl -s https://<адрес>/api/v1/health
# {"ok":true,"queue":0,"workers":3,"busy":0,"analyzer":"real","analyzer_state":"ready"}
```

`"analyzer":"fake"` означает режим‑заглушку (см. [../service/README.md](../service/README.md)); для проверки жюри нужен `real`.

## Как пользоваться интерфейсом

1. Откройте адрес сервиса в браузере.
2. Перетащите видео в рамку или нажмите на неё и выберите файл (.mov или .mp4, до 200 МБ).
3. Страница показывает прогресс загрузки файла, затем текущий этап анализа и место в очереди. Она сама опрашивает сервер; адрес страницы меняется на `/#job=<job_id>`, по этой ссылке результат можно открыть позже.
4. После обработки показывается вердикт: «Нарушение не обнаружено» или «Обнаружены признаки возможных нарушений».
5. Для каждого нарушения — карточка: тип и код (T1…T13), таймкод, уверенность, описание, норма со ссылкой на adilet.zan.kz и кадры‑доказательства. По карточке видео перематывается к таймкоду, кадр можно открыть крупно. Область, о которой идёт речь (стол, сейф, ноутбук, дверь или люди), обведена на кадре красной рамкой.
6. Если анализ завершился ошибкой, страница показывает её причину.

## Быстрый пример API

```bash
URL=https://<адрес>; KEY=<ключ>
curl -X POST "$URL/api/v1/analyze" -H "X-API-Key: $KEY" -F "file=@video_017.mov"
# {"job_id":"99c7837d0400","status":"processing"}
curl -H "X-API-Key: $KEY" "$URL/api/v1/jobs/99c7837d0400"
# {"job_id":"99c7837d0400","status":"processing","progress":0.5,"stage":"...","queue_position":0}
# ... затем {"job_id":"99c7837d0400","status":"done","verdict":"violation","violations":[...],"meta":{...}}
```

Полная схема ответа и коды ошибок — в [API.md](API.md).

## Установка на новый GPU-хост

Так сервис переносится во внутренний контур банка (раздел 7 ТЗ, «Переносимость»). Интернет нужен только на шагах 2–5 (пакеты и веса). После этого обработка видео идёт без сети.

Ниже — пути и пользователь текущего хоста (`shadeform`, код в `~/forte`). Пользователь `shadeform` и путь `/home/shadeform` прописаны в `service/deploy/forte-api.service`, `service/deploy/forte-tunnel.service` и в `DEFAULT_WEIGHTS` в `pipeline/persons.py`. На хосте с другим пользователем эти строки нужно поправить.

### 1. Железо и ОС

- Проверено: NVIDIA L40S 48 ГБ (46 068 МиБ по `nvidia-smi`), драйвер 580.126.09 (CUDA 13.0), 12 vCPU, 70 ГБ ОЗУ, Ubuntu 22.04.5 LTS.
- Видеопамять на этом хосте: сервер vLLM занимает 33.3 ГБ (`--gpu-memory-utilization 0.65`), процесс сервиса с YOLO26x — около 1.3 ГБ (`nvidia-smi`, 18.10.2026).
- Минимум — около 24 ГБ VRAM: веса Qwen3‑VL‑8B в BF16 занимают 17 ГБ на диске, остальное — KV‑кэш vLLM и YOLO26x. На такой карте поднимите долю памяти vLLM (`VLLM_GPU_UTIL=0.85 bash tools/vllm_serve.sh`, оставив 2–3 ГБ под YOLO) и будьте готовы, что меньший KV‑кэш снизит пропускную способность при параллельных задачах. На 24 ГБ мы не проверяли.
- Диск: веса Qwen3‑VL‑8B — 17 ГБ, `yolo26x.pt` — 119 МБ, плюс две виртуальные среды Python.

### 2. Системные пакеты и код

```bash
sudo apt-get install -y ffmpeg tmux curl rsync      # на хосте: ffmpeg 4.4.2 (ffmpeg и ffprobe нужны и API, и конвейеру)
curl -LsSf https://astral.sh/uv/install.sh | sh     # менеджер пакетов uv, ставится в ~/.local/bin
mkdir -p ~/forte/logs ~/det_weights ~/hf
# скопируйте в ~/forte каталоги pipeline/, service/, tools/ (и analysis/gt_team.json для оценки)
```

Для публичного адреса нужен обратный прокси Caddy (на хосте — v2.11.7 в `/usr/local/bin/caddy`, статический бинарник с caddyserver.com) или, как резерв, `cloudflared` (2026.10.1 в `/usr/local/bin`). Внутри контура банка вместо них сервис `127.0.0.1:8000` публикует штатный обратный прокси.

### 3. Среда det-venv

```bash
# API, декодирование, YOLO, правила (версии — как на рабочем хосте)
~/.local/bin/uv venv --python 3.12 ~/det-venv
~/.local/bin/uv pip install --python ~/det-venv/bin/python torch torchvision "ultralytics==8.4.178" \
  fastapi uvicorn python-multipart httpx numpy pillow scipy opencv-python-headless huggingface_hub
```

### 4. Веса (один раз, при доступе в интернет)

```bash
HF_HOME=~/hf ~/det-venv/bin/hf download Qwen/Qwen3-VL-8B-Instruct
curl -sSL -o ~/det_weights/yolo26x.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt
```

Ultralytics при отсутствии файлов пытается докачать их с GitHub даже в офлайн‑режиме, поэтому всё нужное кладётся заранее. На рабочем хосте в `~/.config/Ultralytics/` лежат `Arial.ttf` и `settings.json` с `"sync": false`.

### 5. Среда vllm-venv и сервер VLM

vLLM тянет свою версию torch, поэтому у него отдельная среда. Запускайте после шага 4: сервер стартует в офлайн‑режиме и берёт веса только из `~/hf`.

```bash
bash ~/forte/tools/vllm_install.sh    # создаёт ~/vllm-venv, ставит vllm и запускает сервер в tmux-сессии vllm
curl -s 127.0.0.1:8001/v1/models      # готов, когда отвечает (модель qwen3vl)
```

`tools/vllm_install.sh` запускает `tools/vllm_serve.sh`, только если tmux‑сессии `vllm` ещё нет. После перезагрузки хоста сервер поднимается вручную:

```bash
tmux new-session -d -s vllm "bash ~/forte/tools/vllm_serve.sh 2>&1 | tee -a ~/forte/logs/vllm.log"
```

`tools/vllm_serve.sh` запускает Qwen/Qwen3-VL-8B-Instruct в BF16 на `127.0.0.1:8001` с `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `VLLM_DO_NOT_TRACK=1`. Доля видеопамяти задаётся `VLLM_GPU_UTIL` (по умолчанию 0.65). Один сервер обслуживает все процессы конвейера.

Версии на рабочем хосте: Python 3.12.15; det-venv — torch 2.14.1+cu130, ultralytics 8.4.178, fastapi 0.143.3, uvicorn 0.54.0, httpx 0.28.1, numpy 2.5.4, pillow 13.0.0, scipy 1.18.1, opencv‑python‑headless 5.0.0.93; vllm-venv — vllm 0.31.0, torch 2.13.0+cu130.

### 6. Сервис (systemd)

```bash
bash ~/forte/service/deploy/install.sh --real   # ставит forte-api, forte-tunnel и forte-proxy, включает автозапуск
sudo ufw allow 80/tcp && sudo ufw allow 443/tcp  # порты прокси
curl -s 127.0.0.1:8000/api/v1/health           # "analyzer":"real","analyzer_state":"ready"
curl -s https://216-81-248-201.sslip.io/api/v1/health   # то же через постоянный адрес
cat ~/forte/service/public_url.txt             # резервный адрес quick tunnel
cat ~/forte/service/api_key.txt                # ключ для жюри
```

- `forte-api.service`: `~/det-venv/bin/python -m uvicorn service.app:app --host 127.0.0.1 --port 8000`, `FORTE_WORKERS=3`, офлайн‑флаги `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `YOLO_OFFLINE=1`, `YOLO_AUTOINSTALL=0`, перезапуск при падении.
- `forte-proxy.service`: Caddy по `service/deploy/Caddyfile` — `https://216-81-248-201.sslip.io` (сертификат выпускается автоматически и хранится в `/var/lib/forte-proxy`) и `http://216.81.248.201`, проксирует на `127.0.0.1:8000`, тело запроса до 210 МБ. На другом хосте в `Caddyfile` подставляется его IP. Только прокси, без перезапуска API: `sudo install -m 644 ~/forte/service/deploy/forte-proxy.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now forte-proxy`; после правки `Caddyfile` — `sudo systemctl reload forte-proxy`.
- `forte-tunnel.service`: `service/deploy/tunnel.sh` поднимает `cloudflared tunnel --url http://127.0.0.1:8000` и записывает адрес в `~/forte/service/public_url.txt`.
- Модели загружаются в фоне после старта: пока `analyzer_state` = `loading`, задачи ждут в очереди. Анализатор ждёт сервер VLM до 300 с, иначе `analyzer_state` = `error`.
- Без systemd: `cd ~/forte && PYTHONPATH=$PWD ~/det-venv/bin/python -m uvicorn service.app:app --host 127.0.0.1 --port 8000`.
- Логи: `journalctl -u forte-api -f`, `journalctl -u forte-proxy -f`, `journalctl -u forte-tunnel -f`. Переменные окружения — в [../service/README.md](../service/README.md).

### 7. Оценка на открытых роликах

Видео лежат в `~/forte/videos/X-NN.mov`, разметка команды — в `~/forte/analysis/gt_team.json`.

```bash
cd ~/forte
# полный прогон анализатора, как в сервисе: результаты в ~/forte/cache/results, затем сводка
~/det-venv/bin/python -m pipeline.batch --clips all --threads 2
~/det-venv/bin/python -m pipeline.batch --clips X-15,X-18 --threads 2 --out ~/forte/cache/results
# только правила на сохранённых сигналах (~/forte/cache/{states,persons,checks,facts}), без GPU и VLM, секунды
~/det-venv/bin/python -m pipeline.offline
~/det-venv/bin/python -m pipeline.offline --show    # плюс текст наблюдения каждого найденного нарушения
# повторная оценка готовых результатов
~/det-venv/bin/python -m pipeline.eval ~/forte/cache/results --gt ~/forte/analysis/gt_team.json
```

`pipeline.offline` используется для подбора порогов в `PARAMS` (`pipeline/rules.py`): каждое изменение правила сразу проверяется на всех 42 роликах. Метрики сводки описаны в [LIMITATIONS.md](LIMITATIONS.md#качество-на-открытых-роликах).
