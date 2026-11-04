import glob
import os
from collections import defaultdict
from ultralytics import YOLO

def main():
    cabin_model = YOLO("weights/cabin/peleng_cabin_best.pt")
    truck_model = YOLO("weights/finetuned/peleng_yolo26x_seg_clean_polish_best.pt")

    imgs = sorted(glob.glob("data/datasets/scen_2_raw/images/*.jpg"))
    print(f"Total sliced images in scen_2_raw: {len(imgs)}")

    cam_counts = defaultdict(lambda: {"total": 0, "cabin_det": 0, "truck_det": 0, "high_conf_cabin": 0})

    batch_size = 32
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i : i + batch_size]
        res_truck = truck_model.predict(batch, conf=0.25, verbose=False)
        res_cabin = cabin_model.predict(batch, conf=0.25, verbose=False)

        for path, rt, rc in zip(batch, res_truck, res_cabin):
            cid = int(os.path.basename(path).split("_")[1].replace("cam", ""))
            cam_counts[cid]["total"] += 1
            if len(rt.boxes) > 0:
                cam_counts[cid]["truck_det"] += 1
            if len(rc.boxes) > 0:
                cam_counts[cid]["cabin_det"] += 1
                if float(rc.boxes.conf[0]) >= 0.5:
                    cam_counts[cid]["high_conf_cabin"] += 1

    tot_truck = sum(c["truck_det"] for c in cam_counts.values())
    tot_cabin = sum(c["cabin_det"] for c in cam_counts.values())
    tot_high_conf = sum(c["high_conf_cabin"] for c in cam_counts.values())

    print("\n--- CAMERA-BY-CAMERA BREAKDOWN ---")
    for cid in sorted(cam_counts.keys()):
        c = cam_counts[cid]
        print(
            f"Camera {cid:02d}: {c['total']:3d} frames | "
            f"Truck visible: {c['truck_det']:3d} ({c['truck_det']/c['total']*100:4.1f}%) | "
            f"Cabin detected: {c['cabin_det']:3d} ({c['cabin_det']/c['total']*100:4.1f}%) | "
            f"High-conf (>=0.5): {c['high_conf_cabin']:3d}"
        )

    print("\n--- OVERALL SUMMARY ---")
    print(f"Total extracted frames: {len(imgs)}")
    print(f"Frames with Truck visible: {tot_truck} ({tot_truck/len(imgs)*100:.1f}%)")
    print(f"Frames with Cabin detected (conf>=0.25): {tot_cabin} ({tot_cabin/len(imgs)*100:.1f}%)")
    print(f"Frames with Cabin high confidence (conf>=0.50): {tot_high_conf} ({tot_high_conf/len(imgs)*100:.1f}%)")
    print(f"Pure background frames (no truck): {len(imgs) - tot_truck} ({(len(imgs) - tot_truck)/len(imgs)*100:.1f}%)")

if __name__ == "__main__":
    main()
