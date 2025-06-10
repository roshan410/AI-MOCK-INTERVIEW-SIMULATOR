import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import sounddevice as sd
import queue
import sys
import pyttsx3
import numpy as np
from vosk import Model, KaldiRecognizer
from gpt4all import GPT4All

# --- TTS ---
tts = pyttsx3.init()
def speak(text):
    tts.say(text)
    tts.runAndWait()

# --- STT ---
q = queue.Queue()
samplerate = 16000
vosk_model = Model("vosk-model-en-us-0.22")
rec = KaldiRecognizer(vosk_model, samplerate)

recording = False

user_answers = []  # Collect user answers for evaluation later

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def listen_until_stop():
    final_text = ""
    try:
        with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            while recording:
                if not q.empty():
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        result = rec.Result()
                        text = eval(result).get("text", "")
                        print(f"Interim text: {text}")
                        if text:
                            final_text += " " + text
            # Get final text too
            partial_result = rec.FinalResult()
            partial_text = eval(partial_result).get("text", "")
            if partial_text:
                print(f"Final text: {partial_text}")
                final_text += " " + partial_text
    except Exception as e:
        print(f"Error during listen: {e}")
    return final_text.strip()

# --- LLM ---
llm = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf", model_path="model/")
def chat(user_answer, current_question, role):
    prompt = (
        f"You are a mock interviewer for a {role} role.\n"
        f"The candidate says: '{user_answer}'\n"
        f"Give a natural, conversational, short follow-up or next question."
    )
    return llm.generate(prompt, max_tokens=80).strip()

def evaluate_answers(answers, role):
    answers_text = "\n".join(f"Answer {i+1}: {ans}" for i, ans in enumerate(answers))
    prompt = (
        f"You are a professional interviewer evaluating a mock interview for a {role} position.\n"
        f"Here are the candidate's answers:\n{answers_text}\n"
        f"Give a final evaluation score out of 10 with 1-2 lines of feedback."
    )
    return llm.generate(prompt, max_tokens=100).strip()

# --- GUI App ---
class InterviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mock Interview AI")
        self.root.geometry("800x700")
        self.root.configure(bg="#121212")

        self.interview_active = False
        self.current_question = "Can you please introduce yourself?"
        self.selected_role = tk.StringVar(value="Data Analyst")

        title = tk.Label(root, text="👨‍💼 Mock Interview AI", font=("Helvetica", 20, "bold"), bg="#121212", fg="#ffffff")
        title.pack(pady=10)

        role_frame = tk.Frame(root, bg="#121212")
        role_frame.pack(pady=5)
        tk.Label(role_frame, text="Select Role:", bg="#121212", fg="white", font=("Helvetica", 12)).pack(side=tk.LEFT)
        self.role_menu = ttk.Combobox(role_frame, textvariable=self.selected_role, values=[
            "Data Analyst", "Software Developer", "Product Manager", "Marketing Executive"
        ], font=("Helvetica", 12), state="readonly", width=30)
        self.role_menu.pack(side=tk.LEFT, padx=10)

        self.chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Helvetica", 12), state="disabled",
                                                  bg="#1e1e1e", fg="#ffffff", insertbackground="white", padx=10, pady=10)
        self.chat_box.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(root, bg="#121212")
        input_frame.pack(pady=5, fill=tk.X)

        self.entry = tk.Entry(input_frame, font=("Helvetica", 12), bg="#1e1e1e", fg="white", insertbackground="white")
        self.entry.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)

        self.send_btn = tk.Button(input_frame, text="Send", command=self.send_text, font=("Helvetica", 12),
                                  bg="#ff0050", fg="white", activebackground="#e60045", width=8)
        self.send_btn.pack(side=tk.RIGHT, padx=10)

        controls = tk.Frame(root, bg="#121212")
        controls.pack(pady=10)

        self.start_btn = tk.Button(controls, text="🎤 Start Interview", command=self.toggle_interview,
                                   font=("Helvetica", 12), bg="#4CAF50", fg="white", activebackground="#43a047", width=18)
        self.start_btn.grid(row=0, column=0, padx=10)

        self.rec_btn = tk.Button(controls, text="🎙️ Start Recording", command=self.start_recording,
                                 font=("Helvetica", 12), bg="#ff9800", fg="white", activebackground="#fb8c00", width=18)
        self.rec_btn.grid(row=0, column=1, padx=10)

        self.stop_rec_btn = tk.Button(controls, text="⏹️ Stop Recording", command=self.stop_recording,
                                      font=("Helvetica", 12), bg="#f44336", fg="white", activebackground="#e53935", width=18)
        self.stop_rec_btn.grid(row=0, column=2, padx=10)

        # Greet and ask first question
        self.say_and_display("Interviewer", self.current_question)

    def say_and_display(self, speaker, message):
        self.chat_box.configure(state="normal")
        self.chat_box.insert(tk.END, f"{speaker}: {message}\n\n")
        self.chat_box.configure(state="disabled")
        self.chat_box.yview(tk.END)
        speak(message)

    def toggle_interview(self):
        global user_answers
        self.interview_active = not self.interview_active
        if self.interview_active:
            user_answers = []
            self.start_btn.config(text="⏹ Stop Interview", bg="#f44336")
            self.say_and_display("Interviewer", "Let's begin the mock interview.")
        else:
            self.start_btn.config(text="🎤 Start Interview", bg="#4CAF50")
            self.say_and_display("Interviewer", "Interview stopped. Thank you!")
            role = self.selected_role.get()
            score = evaluate_answers(user_answers, role)
            self.say_and_display("Evaluation", score)

    def send_text(self):
        if not self.interview_active:
            return
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.entry.delete(0, tk.END)
        self.say_and_display("You", user_input)
        user_answers.append(user_input)
        threading.Thread(target=self.respond, args=(user_input,)).start()

    def respond(self, user_input):
        role = self.selected_role.get()
        response = chat(user_input, self.current_question, role)
        self.current_question = response
        self.say_and_display("Interviewer", response)

    def start_recording(self):
        global recording
        if not self.interview_active:
            return
        recording = True
        self.say_and_display("System", "Recording started. Speak now...")
        self.listen_thread = threading.Thread(target=self.listen_and_process)
        self.listen_thread.start()

    def stop_recording(self):
        global recording
        if not self.interview_active:
            return
        recording = False
        self.say_and_display("System", "Recording stopped. Processing...")
        self.root.after(500, lambda: print("Allowing buffer to process..."))

    def listen_and_process(self):
        text = listen_until_stop()
        if text:
            self.say_and_display("You", text)
            user_answers.append(text)
            self.respond(text)

# --- Run App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = InterviewApp(root)
    speak("Welcome to Mock Interview AI. You can type or speak your answers.")
    root.mainloop()