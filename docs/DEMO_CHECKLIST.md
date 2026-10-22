# Mentor Demo Checklist

No computer-vision system is error-free. This checklist configures VisionGuard for conservative live output and reduces avoidable demo failures.

## Tonight

1. Open **Face Registration** and register each person with three clear images: front, slight left, and slight right. Use the same room and lighting planned for the demo. Existing profiles in this repository currently contain only one reference image each.
2. Rehearse the exact objects you plan to show. Extended mode combines standard YOLO11, the fine-tuned daily-object checkpoint, and YOLO-World; camera angle and lighting still matter.
3. Remove hanging clothes, mirrors, and clutter from the camera background. These can resemble COCO classes.
4. Restart the application once after registration and verify every identity and object before the presentation.

## Recommended Live Detection settings

- **Demo Safe Mode:** On
- **Object model:** Diverse for maximum category coverage; Extended if the live frame rate is too low
- **Object Tracking:** On
- **YOLO11 confidence:** 0.45
- **NMS IoU:** 0.50
- **YOLO-World confidence:** 0.02
- **Face Recognition Threshold:** 0.50
- **Capture Screenshots:** Off unless evidence capture is part of the demonstration

Demo Safe Mode displays people and face boxes immediately. Other object labels must agree across two consecutive frames before they are drawn or written to history.

## During the demo

- Use even front lighting and keep the camera steady.
- Present one new object at a time, centered and separated from the hand where possible.
- Hold each object still for at least two processed frames.
- Keep the object approximately 30–80 cm from the camera and avoid strong glare.
- If the Extended model is too slow on the CPU-only machine, use Accurate mode for standard COCO objects and explain that pen/accessory detection is experimental.

## Honest limitation

The local `daily_objects.pt` checkpoint was fine-tuned on a 598-image Open Images subset and evaluated on an untouched 90-image split (`0.649` mAP@0.50; `0.493` mAP@0.50:0.95). Do not describe this as zero error. For the most reliable demo, add labeled examples from the intended webcam, room, lighting, poses, and exact objects before another training run.
