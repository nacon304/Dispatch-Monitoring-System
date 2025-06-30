import streamlit as st
from detection import detect

def main():
    st.title("Dish & Tray Monitoring System")
    inference_msg = st.empty()
    st.sidebar.title("System Configuration")

    st.sidebar.markdown("**🎥 Input source:** Local video")

    conf_thres = float(st.sidebar.text_input(
        "🔍 Detection Confidence Threshold",
        "0.25",
        help="Minimum confidence for accepting a detection result.\nLower values = more detections, but may include false positives."
    ))
    conf_thres_drift = float(st.sidebar.text_input(
        "⚠️ Drift Detection Threshold",
        "0.75",
        help="If an object's confidence falls below this threshold, it's considered a 'drift'.\nUsed to flag potentially inaccurate detections."
    ))

    fps_drop_warn_thresh = 8

    save_output_video = st.sidebar.radio(
        "💾 Save output video?",
        ('Yes', 'No'),
        help="Save a new video with bounding boxes and annotations.\nIf 'No', only live preview is shown."
    )
    nosave = save_output_video != 'Yes'
    display_labels = not nosave

    save_poor_frame = st.sidebar.radio(
        "🧹 Save low-confidence frames?",
        ('Yes', 'No'),
        help="Save frames containing objects with low detection confidence (< Drift Threshold).\nUsed for retraining or error analysis later."
    )
    save_poor_frame__ = save_poor_frame == 'Yes'

    video = st.sidebar.file_uploader("📂 Upload video file", type=["mp4", "avi"])
    start_tracking = st.sidebar.button("🚀 Start Monitoring")

    if start_tracking:
        if video is None:
            st.warning("⚠️ Please upload a video file to start.")
            return

        stframe = st.empty()

        st.subheader("📊 Inference Stats")
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown("**⏱ Frame Rate (FPS)**")
            kpi_fps_text = st.markdown("0")
            fps_warn = st.empty()
        with kpi2:
            st.markdown("**🍽 Objects in Current Frame**")
            kpi_object_text = st.markdown("0")
        with kpi3:
            st.markdown("**📦 Total Objects Detected**")
            kpi_summary_text = st.markdown("0")

        st.subheader("📉 Inference Overview")
        inf_ov_1, inf_ov_2, inf_ov_3, inf_ov_4 = st.columns(4)
        with inf_ov_1:
            st.markdown(f"**🧪 Low Confidence Classes (< {conf_thres_drift})**")
            drift_obj_list_text = st.markdown("0")
        with inf_ov_2:
            st.markdown("**🖼 Poor Performing Frames Count**")
            drift_count_text = st.markdown("0")
        with inf_ov_3:
            st.markdown("**📉 Minimum FPS**")
            min_fps_text = st.markdown("0")
        with inf_ov_4:
            st.markdown("**📈 Maximum FPS**")
            max_fps_text = st.markdown("0")

        st.subheader("💻 System Stats")
        js1, js2, js3 = st.columns(3)
        with js1:
            st.markdown("**🧠 RAM Usage**")
            sys_ram_text = st.markdown("0")
        with js2:
            st.markdown("**🖥 CPU Usage**")
            sys_cpu_text = st.markdown("0")
        with js3:
            st.markdown("**🎮 GPU Memory**")
            sys_gpu_text = st.markdown("0")

        detect(
            source=video.name,
            stframe=stframe,
            kpi_fps_text=kpi_fps_text,
            kpi_object_text=kpi_object_text,
            kpi_summary_text=kpi_summary_text,
            sys_ram_text=sys_ram_text,
            sys_cpu_text=sys_cpu_text,
            sys_gpu_text=sys_gpu_text,
            conf_thres=conf_thres,
            nosave=nosave,
            display_labels=display_labels,
            drift_conf_thres=conf_thres_drift,
            save_drift_frames=save_poor_frame__,
            drift_obj_list_text=drift_obj_list_text,
            drift_count_text=drift_count_text,
            min_fps_text=min_fps_text,
            max_fps_text=max_fps_text,
            fps_warn=fps_warn,
            fps_drop_warn_threshold=fps_drop_warn_thresh,
        )

        inference_msg.success("✅ Inference Complete!")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
