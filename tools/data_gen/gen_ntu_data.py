import os
import zipfile
import glob
import numpy as np
import pickle
import logging
from tqdm import tqdm
from google.colab import drive

# ==========================================
# 1. CẤU HÌNH & GIẢI NÉN (QUAN TRỌNG)
# ==========================================

# Mount Google Drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# --- ĐƯỜNG DẪN FILE ZIP TRÊN DRIVE CỦA BẠN ---
ZIP_FILE_PATH = '/content/drive/MyDrive/Data/nturgbd_skeletons_s001_to_s017.zip'

# --- ĐƯỜNG DẪN GIẢI NÉN TẠM THỜI TRÊN COLAB (Local Runtime) ---
# Dùng ổ cứng của Colab để xử lý cho nhanh, không giải nén ngược lên Drive
LOCAL_EXTRACT_PATH = '/content/temp_ntu_skeletons'

# --- ĐƯỜNG DẪN OUTPUT (LƯU KẾT QUẢ VỀ DRIVE) ---
OUTPUT_PATH = '/content/drive/MyDrive/Data/NTU_SKELETONS_60_V4'

def unzip_dataset(zip_path, extract_to):
    print(f"🚀 Đang giải nén: {zip_path}")
    print(f"📂 Giải nén vào: {extract_to} (Local Runtime)...")

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Lấy danh sách file để hiện thanh loading
        members = zip_ref.infolist()
        # Chỉ giải nén file .skeleton để tiết kiệm thời gian
        members = [m for m in members if m.filename.endswith('.skeleton')]

        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, extract_to)

    print("✅ Giải nén hoàn tất!")

# Thực hiện giải nén
unzip_dataset(ZIP_FILE_PATH, LOCAL_EXTRACT_PATH)

# --- TỰ ĐỘNG TÌM FOLDER CHỨA FILE .SKELETON ---
# Đôi khi file zip có folder lồng nhau, ta cần tìm đúng folder chứa file
INPUT_SKELETON_PATH = LOCAL_EXTRACT_PATH
# Kiểm tra xem có folder con không
sub_dirs = [x[0] for x in os.walk(LOCAL_EXTRACT_PATH)]
for d in sub_dirs:
    if len(glob.glob(os.path.join(d, '*.skeleton'))) > 0:
        INPUT_SKELETON_PATH = d
        break

print(f"🎯 Thư mục chứa skeleton để xử lý: {INPUT_SKELETON_PATH}")

# Tạo thư mục output trên Drive nếu chưa có
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    os.makedirs(os.path.join(OUTPUT_PATH, 'raw_data'))
    os.makedirs(os.path.join(OUTPUT_PATH, 'denoised_data'))
    os.makedirs(os.path.join(OUTPUT_PATH, 'statistics'))

# ==========================================
# 2. CÁC HÀM XỬ LÝ (GIỮ NGUYÊN LOGIC CỦA BẠN)
# ==========================================

def generate_statistics(input_dir, output_dir):
    print("\n--- 1. Đang quét file và tạo Metadata (Statistics) ---")
    ske_files = glob.glob(os.path.join(input_dir, '*.skeleton'))
    ske_names = [os.path.splitext(os.path.basename(f))[0] for f in ske_files]
    ske_names.sort()

    if len(ske_names) == 0:
        raise ValueError(f"❌ Không tìm thấy file .skeleton nào trong {input_dir}")

    print(f"Tìm thấy {len(ske_names)} files.")

    cameras = []
    performers = []
    labels = []

    for name in ske_names:
        # Parse tên file: S001C002P003R002A013
        c_str = name[name.find('C')+1 : name.find('C')+4]
        p_str = name[name.find('P')+1 : name.find('P')+4]
        a_str = name[name.find('A')+1 : name.find('A')+4]

        cameras.append(int(c_str))
        performers.append(int(p_str))
        labels.append(int(a_str))

    stat_path = os.path.join(output_dir, 'statistics')
    np.savetxt(os.path.join(stat_path, 'skes_available_name.txt'), ske_names, fmt='%s')
    np.savetxt(os.path.join(stat_path, 'camera.txt'), np.array(cameras), fmt='%d')
    np.savetxt(os.path.join(stat_path, 'performer.txt'), np.array(performers), fmt='%d')
    np.savetxt(os.path.join(stat_path, 'label.txt'), np.array(labels), fmt='%d')
    print("Đã tạo xong metadata.")

def get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger):
    ske_file = os.path.join(skes_path, ske_name + '.skeleton')
    if not os.path.exists(ske_file):
        return None

    with open(ske_file, 'r') as fr:
        str_data = fr.readlines()

    num_frames = int(str_data[0].strip('\r\n'))
    frames_drop = []
    bodies_data = dict()
    valid_frames = -1
    current_line = 1

    for f in range(num_frames):
        num_bodies = int(str_data[current_line].strip('\r\n'))
        current_line += 1

        if num_bodies == 0:
            frames_drop.append(f)
            continue

        valid_frames += 1
        joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)
        colors = np.zeros((num_bodies, 25, 2), dtype=np.float32)

        for b in range(num_bodies):
            bodyID = str_data[current_line].strip('\r\n').split()[0]
            current_line += 1
            num_joints = int(str_data[current_line].strip('\r\n'))
            current_line += 1

            for j in range(num_joints):
                temp_str = str_data[current_line].strip('\r\n').split()
                joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32)
                colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32)
                current_line += 1

            if bodyID not in bodies_data:
                body_data = dict()
                body_data['joints'] = joints[b]
                body_data['colors'] = colors[b, np.newaxis]
                body_data['interval'] = [valid_frames]
            else:
                body_data = bodies_data[bodyID]
                body_data['joints'] = np.vstack((body_data['joints'], joints[b]))
                body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis]))
                body_data['interval'].append(body_data['interval'][-1] + 1)

            bodies_data[bodyID] = body_data

    num_frames_drop = len(frames_drop)
    if num_frames_drop > 0:
        frames_drop_skes[ske_name] = np.array(frames_drop, dtype=int)
        frames_drop_logger.info(f'{ske_name}: {num_frames_drop} frames missed')

    if len(bodies_data) > 1:
        for body_data in bodies_data.values():
            body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0))

    return {'name': ske_name, 'data': bodies_data, 'num_frames': num_frames - num_frames_drop}

def run_get_raw_skes_data(input_skes_path, output_root_path):
    print("\n--- 2. BẮT ĐẦU: get_raw_skes_data ---")
    stat_path = os.path.join(output_root_path, 'statistics')
    save_raw_path = os.path.join(output_root_path, 'raw_data')
    skes_name_file = os.path.join(stat_path, 'skes_available_name.txt')
    save_data_pkl = os.path.join(save_raw_path, 'raw_skes_data.pkl')

    frames_drop_logger = logging.getLogger('frames_drop')
    frames_drop_logger.setLevel(logging.INFO)
    if frames_drop_logger.hasHandlers(): frames_drop_logger.handlers.clear()
    frames_drop_logger.addHandler(logging.FileHandler(os.path.join(save_raw_path, 'frames_drop.log')))

    frames_drop_skes = dict()
    skes_name = np.loadtxt(skes_name_file, dtype=str)
    num_files = skes_name.size

    raw_skes_data = []
    frames_cnt = np.zeros(num_files, dtype=int)

    # Dùng tqdm để hiện thanh tiến trình thay vì in print dòng
    for idx, ske_name in enumerate(tqdm(skes_name, desc="Reading Raw Skeleton")):
        bodies_data = get_raw_bodies_data(input_skes_path, ske_name, frames_drop_skes, frames_drop_logger)
        if bodies_data is None: continue
        raw_skes_data.append(bodies_data)
        frames_cnt[idx] = bodies_data['num_frames']

    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)

    np.savetxt(os.path.join(save_raw_path, 'frames_cnt.txt'), frames_cnt, fmt='%d')
    print('Saved raw bodies data.')

# --- DENOISING ---
noise_len_thres = 11
noise_spr_thres1 = 0.8
noise_spr_thres2 = 0.69754

def get_one_actor_points(body_data, num_frames):
    joints = np.zeros((num_frames, 75), dtype=np.float32)
    colors = np.ones((num_frames, 1, 25, 2), dtype=np.float32) * np.nan
    start, end = body_data['interval'][0], body_data['interval'][-1]
    joints[start:end + 1] = body_data['joints'].reshape(-1, 75)
    colors[start:end + 1, 0] = body_data['colors']
    return joints, colors

def get_valid_frames_by_spread(points):
    num_frames = points.shape[0]
    valid_frames = []
    for i in range(num_frames):
        x, y = points[i, :, 0], points[i, :, 1]
        if (x.max() - x.min()) <= noise_spr_thres1 * (y.max() - y.min()):
            valid_frames.append(i)
    return valid_frames

def denoising_bodies_data(bodies_data):
    ske_name = bodies_data['name']
    bodies_data = bodies_data['data']

    # By Length
    new_bodies_data = bodies_data.copy()
    for (bodyID, body_data) in list(new_bodies_data.items()):
        if len(body_data['interval']) <= noise_len_thres:
            del bodies_data[bodyID]
    if len(bodies_data) == 1: return list(bodies_data.items())

    # By Spread
    new_bodies_data = bodies_data.copy()
    for (bodyID, body_data) in list(new_bodies_data.items()):
        if len(bodies_data) == 1: break
        valid_frames = get_valid_frames_by_spread(body_data['joints'].reshape(-1, 25, 3))
        num_frames = len(body_data['interval'])
        if num_frames - len(valid_frames) == 0: continue
        ratio = (num_frames - len(valid_frames)) / float(num_frames)
        if ratio >= noise_spr_thres2:
            del bodies_data[bodyID]
        else:
            joints = body_data['joints'].reshape(-1, 25, 3)[valid_frames]
            body_data['motion'] = min(body_data['motion'], np.sum(np.var(joints.reshape(-1, 3), axis=0)))

    if len(bodies_data) == 1: return list(bodies_data.items())

    # Sort by motion
    bodies_motion = {bid: bd['motion'] for bid, bd in bodies_data.items()}
    sorted_bodies = sorted(bodies_motion.items(), key=lambda x: x[1], reverse=True)
    return [(bid, bodies_data[bid]) for bid, _ in sorted_bodies]

def get_two_actors_points(bodies_data):
    num_frames = bodies_data['num_frames']
    bodies_data_list = denoising_bodies_data(bodies_data)
    joints = np.zeros((num_frames, 150), dtype=np.float32)
    colors = np.ones((num_frames, 2, 25, 2), dtype=np.float32) * np.nan

    if len(bodies_data_list) == 1:
        _, body_data = bodies_data_list[0]
        j, c = get_one_actor_points(body_data, num_frames)
        joints[:, :75] = j
        colors[:, 0] = c[:, 0]
    elif len(bodies_data_list) > 1:
        _, actor1 = bodies_data_list[0]
        s1, e1 = actor1['interval'][0], actor1['interval'][-1]
        joints[s1:e1 + 1, :75] = actor1['joints'].reshape(-1, 75)
        colors[s1:e1 + 1, 0] = actor1['colors']

        _, actor2 = bodies_data_list[1]
        s2, e2 = actor2['interval'][0], actor2['interval'][-1]
        joints[s2:e2 + 1, 75:] = actor2['joints'].reshape(-1, 75)
        colors[s2:e2 + 1, 1] = actor2['colors']

    return joints, colors

def run_get_raw_denoised_data(root_path):
    print("\n--- 3. BẮT ĐẦU: get_raw_denoised_data ---")
    raw_data_file = os.path.join(root_path, 'raw_data', 'raw_skes_data.pkl')
    save_denoised_path = os.path.join(root_path, 'denoised_data')
    if not os.path.exists(save_denoised_path): os.makedirs(save_denoised_path)

    with open(raw_data_file, 'rb') as fr:
        raw_skes_data = pickle.load(fr)

    raw_denoised_joints = []
    frames_cnt = []

    for bodies_data in tqdm(raw_skes_data, desc="Denoising"):
        num_bodies = len(bodies_data['data'])
        if num_bodies == 1:
            body_data = list(bodies_data['data'].values())[0]
            joints, _ = get_one_actor_points(body_data, bodies_data['num_frames'])
        else:
            joints, colors = get_two_actors_points(bodies_data)
            # Remove missing
            valid = np.where(joints.sum(axis=1) != 0)[0]
            if len(valid) > 0: joints = joints[valid]

        raw_denoised_joints.append(joints)
        frames_cnt.append(joints.shape[0])

    with open(os.path.join(save_denoised_path, 'raw_denoised_joints.pkl'), 'wb') as f:
        pickle.dump(raw_denoised_joints, f, pickle.HIGHEST_PROTOCOL)

    np.savetxt(os.path.join(save_denoised_path, 'frames_cnt.txt'), np.array(frames_cnt, int), fmt='%d')
    print('Saved denoised data.')

# --- TRANSFORMATION ---
def seq_translation(skes_joints):
    print("\n--- 4.1 Transformation: Centering & Normalization ---")
    for idx, ske_joints in enumerate(tqdm(skes_joints, desc="Translation")):
        if ske_joints.shape[0] == 0: continue
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2

        # Tìm frame hợp lệ đầu tiên để lấy gốc
        i = 0
        while i < ske_joints.shape[0]:
            if np.any(ske_joints[i, :75] != 0): break
            i += 1
        if i >= ske_joints.shape[0]: i = 0

        origin = np.copy(ske_joints[i, 3:6]) # SpineBase

        for f in range(ske_joints.shape[0]):
            if num_bodies == 1: ske_joints[f] -= np.tile(origin, 25)
            else: ske_joints[f] -= np.tile(origin, 50)

        # Normalization
        ske_joints = np.clip(ske_joints, -2.0, 2.0) / 2.0
        skes_joints[idx] = ske_joints
    return skes_joints

def align_frames(skes_joints, frames_cnt):
    print("\n--- 4.2 Transformation: Align Frames ---")
    max_num_frames = min(frames_cnt.max(), 300)
    num_skes = len(skes_joints)
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        if num_frames == 0: continue
        if num_frames > max_num_frames:
            ske_joints = ske_joints[:max_num_frames]
            num_frames = max_num_frames

        if ske_joints.shape[1] == 75:
            aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints, np.zeros_like(ske_joints)))
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints

    return aligned_skes_joints

def run_seq_transformation(root_path):
    print("\n--- 4. BẮT ĐẦU: seq_transformation ---")
    stat_path = os.path.join(root_path, 'statistics')
    denoised_path = os.path.join(root_path, 'denoised_data')

    camera = np.loadtxt(os.path.join(stat_path, 'camera.txt'), dtype=int)
    performer = np.loadtxt(os.path.join(stat_path, 'performer.txt'), dtype=int)
    label = np.loadtxt(os.path.join(stat_path, 'label.txt'), dtype=int) - 1
    frames_cnt = np.loadtxt(os.path.join(denoised_path, 'frames_cnt.txt'), dtype=int)

    with open(os.path.join(denoised_path, 'raw_denoised_joints.pkl'), 'rb') as fr:
        skes_joints = pickle.load(fr)

    skes_joints = seq_translation(skes_joints)
    skes_joints = align_frames(skes_joints, frames_cnt)

    # Split train/test
    def one_hot_vector(l):
        res = np.zeros((len(l), 60))
        for i, val in enumerate(l): res[i, val] = 1
        return res

    # Evaluation Splits
    for evaluation in ['CS', 'CV']:
        print(f"Generating {evaluation} split...")
        train_indices, test_indices = np.empty(0, int), np.empty(0, int)

        if evaluation == 'CS':
            train_ids = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
            test_ids = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]
            for i in train_ids: train_indices = np.hstack((train_indices, np.where(performer == i)[0]))
            for i in test_ids: test_indices = np.hstack((test_indices, np.where(performer == i)[0]))
        else:
            train_ids, test_ids = [2, 3], [1]
            for i in train_ids: train_indices = np.hstack((train_indices, np.where(camera == i)[0]))
            for i in test_ids: test_indices = np.hstack((test_indices, np.where(camera == i)[0]))

        # Filter valid indices
        max_idx = len(skes_joints)
        train_indices = train_indices[train_indices < max_idx]
        test_indices = test_indices[test_indices < max_idx]

        save_name = os.path.join(root_path, f'NTU60_{evaluation}.npz')
        np.savez(save_name,
                 x_train=skes_joints[train_indices], y_train=one_hot_vector(label[train_indices]),
                 x_test=skes_joints[test_indices], y_test=one_hot_vector(label[test_indices]))
        print(f"✅ Saved: {save_name}")

# ==========================================
# 3. CHẠY TOÀN BỘ QUY TRÌNH
# ==========================================
try:
    generate_statistics(INPUT_SKELETON_PATH, OUTPUT_PATH)
    run_get_raw_skes_data(INPUT_SKELETON_PATH, OUTPUT_PATH)
    run_get_raw_denoised_data(OUTPUT_PATH)
    run_seq_transformation(OUTPUT_PATH)
    print("\n🎉 HOÀN THÀNH TẤT CẢ! Kiểm tra folder NTU_SKELETONS_60_V3 trên Drive.")
except Exception as e:
    print(f"\n❌ LỖI: {e}")
    import traceback
    traceback.print_exc()