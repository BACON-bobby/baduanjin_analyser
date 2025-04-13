import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import av
import os

def cv2_put_text(img, text, pos, font_path, font_size, color):
    """使用PIL在OpenCV图像上绘制中文"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, font_size)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"文字绘制错误: {e}")
        # 如果字体加载失败，返回原图
        return img

class BaduanjinWeb(VideoTransformerBase):
    def __init__(self):
        # 初始化MediaPipe组件
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        
        # 样式配置
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3)
        self.hand_drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
        
        # 需要隐藏的姿势手部关键点
        self.HIDDEN_POSE_HAND_LANDMARKS = [
            self.mp_pose.PoseLandmark.LEFT_PINKY,
            self.mp_pose.PoseLandmark.RIGHT_PINKY,
            self.mp_pose.PoseLandmark.LEFT_INDEX,
            self.mp_pose.PoseLandmark.RIGHT_INDEX,
            self.mp_pose.PoseLandmark.LEFT_THUMB,
            self.mp_pose.PoseLandmark.RIGHT_THUMB,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        
        self.current_action = 1
        self.show_list = True
        self.font_path = self._get_font_path("STXINGKA.TTF")
        self.action_names = {
            1: "1. 双手托天理三焦",
            2: "2. 左右开弓似射雕",
            3: "3. 调理脾胃须单举",
            4: "4. 五劳七伤往后瞧",
            5: "5. 摇头摆尾去心火",
            6: "6. 两手攀足固肾腰",
            7: "7. 攒拳怒目增气力",
            8: "8. 背后七颠百病消"
        }

        # 初始化模型
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,  # 必须添加此参数(静态图像模式)
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            model_complexity=1
        )

    def _get_font_path(self, font_name):
        """确保优先从本地fonts目录加载字体"""
        # 1. 检查本地fonts目录
        local_font_path = os.path.join(os.path.dirname(__file__), "fonts", font_name)
        if os.path.exists(local_font_path):
            print(f"✅ 使用本地字体：{local_font_path}")
            return local_font_path
        
        # 2. 检查系统字体目录
        system_font_paths = {
            'Windows': f"C:/Windows/Fonts/{font_name}",
            'Darwin': f"/System/Library/Fonts/{font_name}",
            'Linux': f"/usr/share/fonts/truetype/{font_name}"
        }
        sys_path = system_font_paths.get(os.name)
        if sys_path and os.path.exists(sys_path):
            print(f"✅ 使用系统字体：{sys_path}")
            return sys_path
        
        # 3. 保底方案：使用PIL默认字体（可能显示方框）
        print("⚠️ 警告：使用备用字体，中文可能显示异常")
        return ImageFont.load_default().path

    def calculate_angle(self, a, b, c):
        """计算三点之间的角度"""
        a = (a.x, a.y)
        b = (b.x, b.y)
        c = (c.x, c.y)

        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        dot_product = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

        angle = math.degrees(math.acos(dot_product / (mag_ba * mag_bc + 1e-10)))
        return angle

    # 完整保留所有动作检测方法
    def check_action_1(self, landmarks):
        """双手托天"""
        ls = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        le = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        re = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        lw = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        rw = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        left_angle = self.calculate_angle(ls, le, lw)
        right_angle = self.calculate_angle(rs, re, rw)
        return (lw.y < ls.y) and (rw.y < rs.y) and (left_angle > 160) and (right_angle > 160), left_angle, right_angle

    def check_action_2(self, landmarks):
        """左右开弓"""
        ls = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        rh = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        angle = self.calculate_angle(ls, lh, rh)
        return 80 < angle < 120, angle

    def check_action_3(self, landmarks):
        """调理脾胃"""
        lw = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        rw = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        ls = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        return (lw.y < ls.y) and (rw.y > rs.y), None

    def check_action_4(self, landmarks):
        """五劳七伤"""
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        lh = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        rh = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        angle = self.calculate_angle(nose, lh, rh)
        return angle > 45, angle

    def check_action_5(self, landmarks):
        """摇头摆尾"""
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        mid_hip = (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x +
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
        return abs(nose.x - mid_hip) > 0.2, None

    def check_action_6(self, landmarks):
        """两手攀足"""
        lw = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        rw = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        la = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        ra = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        return (lw.y > la.y) and (rw.y > ra.y), None

    def check_action_7(self, landmarks):
        """攒拳怒目"""
        lh = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        rh = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        angle = self.calculate_angle(lh, rh, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE])
        return angle < 100, angle

    def check_action_8(self, landmarks):
        """背后七颠"""
        lh = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL]
        rh = landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL]
        return (lh.y < landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y * 0.95 and
                rh.y < landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y * 0.95), None

    def _draw_hand_connections(self, image, results):
        """绘制手部关键点之间的连接"""
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.hand_drawing_spec,
                self.connection_drawing_spec
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.hand_drawing_spec,
                self.connection_drawing_spec
            )

    def _draw_custom_landmarks(self, image, results):
        """按照指定方式绘制骨骼点"""
        visible_indices = []
        visible_landmarks = []
        original_to_visible = {}
        
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx not in [lm.value for lm in self.HIDDEN_POSE_HAND_LANDMARKS]:
                original_to_visible[idx] = len(visible_indices)
                visible_indices.append(idx)
                visible_landmarks.append(landmark)
        
        visible_connections = []
        for connection in self.mp_holistic.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx in original_to_visible and end_idx in original_to_visible:
                visible_connections.append(
                    (original_to_visible[start_idx], original_to_visible[end_idx]))
        
        class VisibleLandmarkList:
            def __init__(self, landmarks):
                self.landmark = landmarks
                self.visibility = [1.0] * len(landmarks)
        
        visible_landmark_list = VisibleLandmarkList(visible_landmarks)
        
        self.mp_drawing.draw_landmarks(
            image,
            visible_landmark_list,
            visible_connections,
            self.landmark_drawing_spec,
            self.connection_drawing_spec
        )

        h, w = image.shape[:2]
        if (results.left_hand_landmarks and 
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value in original_to_visible):
            hand_root = results.left_hand_landmarks.landmark[0]
            pose_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            cv2.line(image, 
                    (int(pose_elbow.x * w), int(pose_elbow.y * h)),
                    (int(hand_root.x * w), int(hand_root.y * h)),
                    self.connection_drawing_spec.color,
                    self.connection_drawing_spec.thickness)
        
        if (results.right_hand_landmarks and 
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value in original_to_visible):
            hand_root = results.right_hand_landmarks.landmark[0]
            pose_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            cv2.line(image, 
                    (int(pose_elbow.x * w), int(pose_elbow.y * h)),
                    (int(hand_root.x * w), int(hand_root.y * h)),
                    self.connection_drawing_spec.color,
                    self.connection_drawing_spec.thickness)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        results = self.holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            self._draw_custom_landmarks(img, results)
            self._draw_hand_connections(img, results)
        
        feedback = "等待检测中..."
        if results.pose_landmarks:
            try:
                check_method = getattr(self, f"check_action_{self.current_action}")
                is_correct, *angles = check_method(results.pose_landmarks.landmark)
                feedback = "动作正确!" if is_correct else "请调整姿势"
                if angles:
                    feedback += f" ({' | '.join([f'{x:.1f}°' for x in angles if x is not None])})"
            except Exception as e:
                feedback = f"检测错误: {str(e)}"
        
        img = self.draw_interface(img, feedback)
        return img

    def draw_interface(self, image, feedback):
        h, w = image.shape[:2]
        
        if self.show_list:
            panel = np.zeros((h, 300, 3), dtype=np.uint8)
            panel[:] = (50, 50, 50)
            
            panel = cv2_put_text(panel, "八段锦动作列表", (20, 30),
                                self.font_path, 30, (0, 255, 255))
            y = 70
            for idx in range(1, 9):
                color = (0, 255, 0) if idx == self.current_action else (200, 200, 200)
                panel = cv2_put_text(panel, self.action_names[idx], (20, y),
                                    self.font_path, 25, color)
                y += 40
            
            combined = np.hstack((panel, image))
            feedback_pos = (320, 50)
        else:
            combined = image
            feedback_pos = (20, 50)
        
        combined = cv2_put_text(combined, feedback, feedback_pos,
                              self.font_path, 20, (255, 255, 0))
        
        combined = cv2_put_text(combined, "空格:切换列表 1-8:选择动作 ESC:退出", 
                              (20 if self.show_list else 320, h-30),
                              self.font_path, 20, (200, 200, 200))
        
        return combined

def main():
    st.set_page_config(
        page_title="八段锦动作分析系统",
        page_icon="🧘",
        layout="wide"
    )
    
    st.title("🏮 八段锦实时动作分析系统")
    st.markdown("### 挑战杯参赛作品 - 基于计算机视觉的传统养生功法分析系统")
    
    ctx = webrtc_streamer(
        key="baduanjin",
        video_processor_factory=BaduanjinWeb,
        async_processing=True,
        media_stream_constraints={
            "video": {"width": 640, "height": 480},
            "audio": False
        }
    )
    
    if ctx.video_processor:
        if st.sidebar.checkbox("显示动作列表"):
            ctx.video_processor.show_list = True
        else:
            ctx.video_processor.show_list = False
            
        action = st.sidebar.selectbox(
            "选择分析动作",
            options=list(range(1, 9)),
            format_func=lambda x: [
                "双手托天理三焦",
                "左右开弓似射雕",
                "调理脾胃须单举",
                "五劳七伤往后瞧",
                "摇头摆尾去心火",
                "两手攀足固肾腰",
                "攒拳怒目增气力",
                "背后七颠百病消"
            ][x-1]
        )
        ctx.video_processor.current_action = action

if __name__ == "__main__":
    main()
