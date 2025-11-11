import os
import gradio as gr
import pyttsx3
from duckduckgo_search import DDGS
from ultralytics import YOLO

# ============================================================
# 🗣️ 1. Reliable Voice Output (restart engine each time)
# ============================================================
def speak(text):
    """Speaks the given text reliably every time."""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print("Speech error:", e)

# ============================================================
# 🌐 2. Smart Web Search (with audio summary)
# ============================================================
def smart_search(query):
    """Perform web search and speak results aloud."""
    if not query.strip():
        speak("Please type something to search.")
        return "Please type something to search."

    results_text = ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if results:
            results_text = "\n\n".join(
                [f"🔹 {r['title']}\n{r['body']}\n🌐 {r['href']}" for r in results]
            )
            summary = results[0]['body']
            speak(f"Here are the top results for {query}. {summary}")
        else:
            results_text = "No results found."
            speak("Sorry, I could not find any results.")
    except Exception as e:
        results_text = f"⚠️ Search error: {e}"
        speak("Sorry, there was a problem connecting to the internet.")
    return results_text

# ============================================================
# 🎥 3. YOLOv8 Object Detection (with voice feedback)
# ============================================================
model = YOLO("yolov8n.pt")

def detect_objects(frame):
    """Detect objects using YOLO and announce detections."""
    result = model(frame)
    names = model.names
    detected_ids = result[0].boxes.cls.tolist() if result[0].boxes is not None else []
    detected_labels = [names[int(i)] for i in detected_ids] if detected_ids else []

    if detected_labels:
        unique = list(set(detected_labels))
        speak(f"I see {', '.join(unique)} in front of you.")
    return result[0].plot()

# ============================================================
# 🎨 4. Improved Gradio UI (Updated Header)
# ============================================================
with gr.Blocks(theme=gr.themes.Soft(), title="GuideLight — AI Assistant for the Visually Impaired") as demo:
    # Header
    gr.HTML(
        """
        <div style='text-align:center; margin-bottom:20px;'>
            <h2 style='color:#007bff;'>💡 GuideLight — Smart Voice & Object Detection Assistant</h2>
            <p>An AI-powered live object detection and voice guide for the visually impaired.</p>
        </div>
        """
    )

    # Tabs
    with gr.Tabs():
        # ---------------- Voice Search Tab ----------------
        with gr.TabItem("🔍 Voice Search Assistant"):
            with gr.Row():
                query_box = gr.Textbox(label="Ask or Search Anything", placeholder="Type your question here...")
                search_button = gr.Button("Search the Web 🌐", elem_id="search-btn")
            search_output = gr.Textbox(label="Search Results", lines=10)
            search_button.click(fn=smart_search, inputs=query_box, outputs=search_output)

        # ---------------- Live Object Detection Tab ----------------
        with gr.TabItem("🎥 Live Object Detection"):
            with gr.Row():
                cam = gr.Image(sources=["webcam"], streaming=True, label="Live Camera Feed")
                live_output = gr.Image(label="Detected Objects")
            detected_list = gr.Textbox(label="Detected Objects (Text Summary)", interactive=False)

            # Stream detection
            def detect_with_list(frame):
                img = detect_objects(frame)
                result = model(frame)
                names = model.names
                detected_ids = result[0].boxes.cls.tolist() if result[0].boxes is not None else []
                detected_labels = [names[int(i)] for i in detected_ids] if detected_ids else []
                detected_summary = ", ".join(list(set(detected_labels))) if detected_labels else "Nothing detected"
                return img, detected_summary

            cam.stream(detect_with_list, inputs=cam, outputs=[live_output, detected_list])

    # Footer
    gr.HTML(
        """
        <div style='text-align:center; margin-top:20px; font-size:12px; color:gray;'>
            © 2025 GuideLight AI | Designed by Chirag Khodiyar
        </div>
        """
    )

# ============================================================
# 🚀 5. Launch App
# ============================================================
demo.launch()
