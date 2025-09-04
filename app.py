from flask import Flask, render_template, Response, request, send_file, jsonify, redirect, url_for, flash
import cv2
import os
import numpy as np
import pandas as pd
import json
import math
import io
import base64

import config
from utils.query_processing import Translation
from utils.faiss import Myfaiss
from utils.mapping import keyframe_to_video_frame
from query.query_encoder import encode_long_query, extract_keywords, encode_hybrid



# Q&A suggest (tùy chọn, nặng)
try:
    from qa_modules import suggest_answer_bundle  # return {'ocr':..., 'count':..., 'caption':..., 'hints':[...] }
except Exception:
    suggest_answer_bundle = None

# TRAKE alignment (tùy chọn)
try:
    from trake_module import align_events
except Exception:
    align_events = None

from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.secret_key = "aic2025-secret"

######## LOAD IMAGE MAP từ map_idx_clean ########
from pathlib import Path

DictImagePath = {}
maps_dir = Path(config.MAP_JSON_DIR)
img_counter = 0

for map_file in sorted(maps_dir.glob("map_*.json")):
    video_id = map_file.stem.replace("map_", "")
    with open(map_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # sort theo frame id
    items = sorted(raw.items(), key=lambda kv: int(kv[1]))
    for fname, frame_idx in items:
        rel_path = f"{video_id}/{fname}"
        DictImagePath[img_counter] = rel_path
        img_counter += 1

LenDictPath = len(DictImagePath)
print(f" Loaded {LenDictPath} images from {len(list(maps_dir.glob('map_*.json')))} map files.")


######## FAISS ########
device = "cuda" if getattr(config, "USE_GPU", False) else "cpu"
bin_file = getattr(config, "FAISS_INDEX_BIN", "faiss_normal_ViT.bin")
features_all_path = getattr(config, "FEATURES_ALL_NPY", "features_all.npy")
MyFaiss = Myfaiss(bin_file, DictImagePath, device, Translation(), "ViT-B/32", features_path=features_all_path)

######## IN-MEMORY RESULTS ########
# Mỗi phần tử:
#   'rid'      : ID duy nhất (ổn định để xoá/kéo-thả)
#   'query_id' : số thứ tự 1..N (được đánh lại sau mọi thay đổi)
#   'type'     : 'KIS'|'QA'|'TRAKE'
#   'video'    : 'Lxx_Vxxx.mp4'
#   'frame'    : int (event1)
#   'answer'   : str (QA) hoặc ''
#   'events'   : [f2, f3, f4, ...] (TRAKE)
RESULTS = []
RID_COUNTER = 1  # chỉ tăng

# Query grouping: all selections from the same text query share one query_id
CURRENT_QUERY_TEXT = ""
CURRENT_QUERY_ID = 0
NEXT_QUERY_ID = 1

def _resolve_abs_keyframe_path(rel_path: str) -> str:
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(config.KEYFRAMES_DIR, rel_path).replace("\\", "/")

def _recompute_query_ids():
    """No-op: giữ nguyên query_id để nhóm theo text query gốc."""
    return

@app.route('/')
@app.route('/home')
def home():
    data = {
        'pagefile': [],
        'num_page': 1,
        'query': '',
        'query_type': 'KIS',
        'is_qa': False,
        'answer': ''
    }
    _recompute_query_ids()
    return render_template('home.html', data=data, results=RESULTS)

@app.route('/textsearch', methods=['GET', 'POST'])
def text_search():
    if request.method == 'GET':
        text_query = (request.args.get('textquery') or '').strip()
        query_type = (request.args.get('query_type') or 'KIS').strip().upper()
        is_qa = (query_type == 'QA') or ((request.args.get('is_qa') or '') in ['1','true','True','on'])
        answer = (request.args.get('answer') or '').strip()
        k = int(request.args.get('k', 10))
    else:
        text_query = (request.form.get('textquery') or '').strip()
        query_type = (request.form.get('query_type') or 'KIS').strip().upper()
        is_qa = (request.form.get('is_qa') == 'on') or (query_type == 'QA')
        answer = (request.form.get('answer') or '').strip()
        try:
            k = int(request.form.get('k', 10))
        except:
            k = 10

    k = max(1, min(200, k))

    if not text_query:
        flash("Vui lòng nhập mô tả truy vấn.")
        return redirect(url_for('home'))

    # Gán/giữ query_id theo text_query
    global CURRENT_QUERY_TEXT, CURRENT_QUERY_ID, NEXT_QUERY_ID
    if text_query != CURRENT_QUERY_TEXT:
        CURRENT_QUERY_TEXT = text_query
        CURRENT_QUERY_ID = NEXT_QUERY_ID
        NEXT_QUERY_ID += 1

    # --- Xử lý query bằng query_encoder ---
    query_vec = encode_long_query(text_query, MyFaiss.model, device)

    if query_vec is None:
        flash("Không thể encode query, vui lòng nhập lại mô tả ngắn gọn hơn.")
        return redirect(url_for('home'))

    #  Thêm log debug
    print("🔎 Query vector shape:", query_vec.shape)
    print("🔎 Query vector norm:", np.linalg.norm(query_vec))

    query_vec = query_vec.astype(np.float32)  # ép kiểu chuẩn cho FAISS

    # --- Search FAISS bằng vector đã encode ---
    D, I, _, list_image_paths = MyFaiss.vector_search(query_vec, k=k)
    list_ids = I

    pagefile = [{'imgpath': p, 'id': int(i)} for p, i in zip(list_image_paths, list_ids)]
    imgperindex = 100
    data = {
        'num_page': max(1, math.ceil(len(pagefile)/imgperindex)),
        'pagefile': pagefile,
        'query': text_query,
        'query_type': query_type,
        'is_qa': is_qa,
        'answer': answer
    }
    _recompute_query_ids()
    return render_template('home.html', data=data, results=RESULTS)

@app.route('/confirm_select', methods=['POST'])
def confirm_select():
    img_id = int(request.form.get('imgid'))
    text_query = request.form.get('textquery', '')
    query_type = (request.form.get('query_type') or 'KIS').strip().upper()
    is_qa = (request.form.get('is_qa') == 'on') or (query_type == 'QA')
    answer = (request.form.get('answer') or '').strip()

    rel_path = DictImagePath.get(img_id, '')
    abs_path = _resolve_abs_keyframe_path(rel_path)

    # Map sang (video, frame)
    map_res = keyframe_to_video_frame(rel_path or '')
    if not map_res:
        flash("Không tìm thấy map keyframe → video cho ảnh đã chọn.")
        return redirect(url_for('home'))
    video_name, frame_idx = map_res

    payload = {
        'img_id': img_id,
        'rel_path': rel_path,
        'abs_path': abs_path,
        'video': video_name,
        'frame': frame_idx,
        'textquery': text_query,
        'query_type': query_type,
        'is_qa': is_qa,
        'answer': answer
    }

    # Gợi ý Q&A (tắt mặc định để không bị chậm)
    qa_enabled = getattr(config, "QA_SUGGEST_ENABLED", False)
    qa_suggest = {}
    if is_qa and qa_enabled and suggest_answer_bundle:
        try:
            qa_suggest = suggest_answer_bundle(abs_path, text_query) or {}
        except Exception as e:
            qa_suggest = {'error': str(e)}
    payload['qa_suggest'] = qa_suggest

    return render_template('confirm.html', payload=payload)

@app.route('/finalize_select', methods=['POST'])
def finalize_select():
    global RID_COUNTER
    query_type = (request.form.get('query_type') or 'KIS').strip().upper()
    video_name = request.form.get('video')
    frame_idx  = int(request.form.get('frame'))
    answer     = (request.form.get('answer') or '').strip()
    text_query = (request.form.get('textquery') or '').strip()

    record = {
        'rid'   : RID_COUNTER,  # cố định
        'query_id': CURRENT_QUERY_ID,
        'type'  : query_type,
        'video' : video_name,
        'frame' : frame_idx,
        'answer': answer if query_type == 'QA' else '',
        'events': []
    }

    # TRAKE (tuỳ chọn)
    if query_type == 'TRAKE' and align_events:
        try:
            # Force DTW when user selects TRAKE; if data missing, module will fallback internally
            frames = align_events(text_query, video_name, MyFaiss, DictImagePath, k_per_event=200, force_dtw=True)
            if frames:
                record['frame']  = frames[0]
                record['events'] = frames[1:4]
        except Exception:
            pass

    RESULTS.append(record)
    RID_COUNTER += 1
    _recompute_query_ids()

    flash(f"Đã thêm kết quả: [{query_type}] {record['video']} @ frame {record['frame']}" +
          (f" | answer='{record['answer']}'" if query_type=='QA' else ""))
    return redirect(url_for('home'))

@app.post('/finalize_multi')
def finalize_multi():
    global RID_COUNTER
    query_type = (request.form.get('query_type') or 'KIS').strip().upper()
    text_query = (request.form.get('textquery') or '').strip()
    img_ids = request.form.getlist('imgids')

    if query_type != 'KIS':
        flash('Multi-select hiện chỉ hỗ trợ KIS.')
        return redirect(url_for('home'))

    if not img_ids:
        flash('Chưa chọn ảnh nào.')
        return redirect(url_for('home'))

    added = 0
    for img_id_str in img_ids:
        try:
            img_id = int(img_id_str)
        except Exception:
            continue

        rel_path = DictImagePath.get(img_id, '')
        if not rel_path:
            continue
        abs_path = _resolve_abs_keyframe_path(rel_path)
        map_res = keyframe_to_video_frame(rel_path or '')
        if not map_res:
            continue
        video_name, frame_idx = map_res

        record = {
            'rid'   : RID_COUNTER,
            'query_id': CURRENT_QUERY_ID,
            'type'  : query_type,  # KIS
            'video' : video_name,
            'frame' : frame_idx,
            'answer': '',
            'events': []
        }
        RESULTS.append(record)
        RID_COUNTER += 1
        added += 1

    _recompute_query_ids()
    flash(f'Đã thêm {added} ảnh cho KIS.' if added else 'Không thêm được ảnh nào.')
    return redirect(url_for('home'))

@app.route('/results')
def list_results():
    _recompute_query_ids()
    return render_template('home.html', data={'pagefile': [], 'num_page': 1, 'query': ''}, results=RESULTS)

@app.route('/export_csv')
def export_csv():
    if not RESULTS:
        flash("Chưa có kết quả để xuất.")
        return redirect(url_for('home'))

    rows = []
    for r in RESULTS:
        video = r.get('video', '').replace('.mp4', '')
        query_type = r.get('type', 'KIS')

        if query_type == 'KIS':
            rows.append(f"{video},{r['frame']}")

        elif query_type == 'QA':
            ans = r.get('answer', '')
            # Nếu answer có dấu phẩy, ngoặc kép hoặc xuống dòng thì phải bao trong ""
            if any(c in ans for c in [',', '"', '\n']):
                ans = '"' + ans.replace('"', '""') + '"'
            rows.append(f"{video},{r['frame']},{ans}")

        elif query_type == 'TRAKE':
            frames = [str(r['frame'])] + [str(f) for f in r.get('events', [])]
            rows.append(f"{video}," + ",".join(frames))

    # Ghép thành text CSV
    csv_content = "\n".join(rows)

    buffer = io.BytesIO()
    buffer.write(csv_content.encode("utf-8"))
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="submission.csv",
        mimetype="text/csv"
    )


@app.post('/clear_results')
def clear_results():
    global RID_COUNTER
    RESULTS.clear()
    RID_COUNTER = 1
    _recompute_query_ids()
    flash("Đã xoá danh sách tạm.")
    return redirect(url_for('home'))

@app.post('/delete_result/<int:rid>')
def delete_result(rid: int):
    before = len(RESULTS)
    RESULTS[:] = [r for r in RESULTS if r.get('rid') != rid]
    _recompute_query_ids()
    flash(f"Đã xoá truy vấn." if len(RESULTS) < before else "Không tìm thấy truy vấn.")
    return redirect(url_for('home'))

@app.post('/reorder_results')
def reorder_results():
    """Nhận mảng rid theo thứ tự mới."""
    try:
        data = request.get_json(force=True) or {}
        order = data.get('order', [])
        id2item = {r['rid']: r for r in RESULTS}
        new_list = []
        for rid in order:
            item = id2item.pop(rid, None)
            if item is not None:
                new_list.append(item)
        new_list.extend(id2item.values())  # phần còn lại
        RESULTS[:] = new_list
        _recompute_query_ids()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400

##############################################
# Legacy routes giữ nguyên
##############################################
@app.route('/imgsearch')
def image_search():
    pagefile = []
    id_query = int(request.args.get('imgid'))
    _, list_ids, _, list_image_paths = MyFaiss.image_search(id_query, k=50)
    imgperindex = 100
    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})
    data = {'num_page': max(1, math.ceil(LenDictPath / imgperindex)), 'pagefile': pagefile}
    _recompute_query_ids()
    return render_template('home.html', data=data, results=RESULTS)

@app.route('/uploadsearch', methods=['POST'])
def upload_search():
    k = int(request.form.get('k', 50))
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400
    pil_img = Image.open(file.stream).convert("RGB")
    _, list_ids, _, list_image_paths = MyFaiss.image_search_from_pil(pil_img, k=k)
    pagefile = [{'imgpath': p, 'id': int(i)} for p, i in zip(list_image_paths, list_ids)]
    imgperindex = 100
    data = {'num_page': max(1, math.ceil(len(pagefile)/imgperindex)), 'pagefile': pagefile, 'query': '[image]'}
    _recompute_query_ids()
    return render_template('home.html', data=data, results=RESULTS)

@app.route('/sketchsearch', methods=['POST'])
def sketch_search():
    k = int(request.form.get('k', 50))
    data_url = request.form.get('sketch_data', '')
    if not data_url.startswith('data:image'):
        return jsonify({'error': 'Invalid sketch data'}), 400
    header, b64 = data_url.split(',', 1)
    img_bytes = base64.b64decode(b64)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    _, list_ids, _, list_image_paths = MyFaiss.image_search_from_pil(pil_img, k=k)
    pagefile = [{'imgpath': p, 'id': int(i)} for p, i in zip(list_image_paths, list_ids)]
    imgperindex = 100
    data = {'num_page': max(1, math.ceil(len(pagefile)/imgperindex)), 'pagefile': pagefile, 'query': '[sketch]'}
    _recompute_query_ids()
    return render_template('home.html', data=data, results=RESULTS)

@app.route('/get_img')
def get_img():
    fpath_in = (request.args.get('fpath') or "").replace("\\", "/").strip()

    # Nếu fpath_in là số id (ví dụ imgid=1234) -> tra từ DictImagePath
    if fpath_in.isdigit():
        rel_path = DictImagePath.get(int(fpath_in), "")
    else:
        rel_path = fpath_in

    # build absolute path từ rel_path
    abs_path = os.path.join(config.KEYFRAMES_DIR, rel_path)

    if os.path.exists(abs_path):
        img = cv2.imread(abs_path)
        list_image_name = abs_path.replace("\\", "/").split("/")
        image_name = "/".join(list_image_name[-2:])
    else:
        print(f"[WARN] Not found: {abs_path}")
        not_found = os.path.join("static", "images", "404.jpg")
        img = cv2.imread(not_found) if os.path.exists(not_found) else np.zeros((720, 1280, 3), dtype=np.uint8)
        image_name = "404.jpg"

    img = cv2.resize(img, (1280, 720))
    img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4, cv2.LINE_AA)
    ret, jpeg = cv2.imencode('.jpg', img)
    return Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
