import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import os

class EmailClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Email Classifier")

        self.email_path = tk.StringVar()
        self.processed_txt_path = None
        self.model_choice = tk.StringVar(value="nb")

        # File input
        tk.Label(root, text="Select .eml or .txt file:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(root, textvariable=self.email_path, width=50).grid(row=0, column=1, padx=10)
        tk.Button(root, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)

        # Model selection
        tk.Label(root, text="Model:").grid(row=1, column=0, sticky="w", padx=10)
        ttk.Combobox(root, textvariable=self.model_choice, values=["nb", "svm", "baseline"], state="readonly", width=10).grid(row=1, column=1, sticky="w", padx=10)

        # Email content viewer with scrollbars
        tk.Label(root, text="Email Content:").grid(row=2, column=0, sticky="nw", padx=10)
        email_frame = tk.Frame(root)
        email_frame.grid(row=2, column=1, columnspan=2, padx=10, pady=5, sticky="nsew")

        h_scroll = tk.Scrollbar(email_frame, orient="horizontal")
        h_scroll.pack(side="bottom", fill="x")

        self.email_viewer = tk.Text(
            email_frame, height=15, width=80,
            bg="white", fg="black",
            wrap="word",  # Enable word wrapping
            xscrollcommand=h_scroll.set,
            state="disabled"
        )
        self.email_viewer.pack(side="left", fill="both", expand=True)

        h_scroll.config(command=self.email_viewer.xview)

        # Run button
        tk.Button(root, text="Run Prediction", command=self.run_pipeline).grid(row=3, column=1, pady=10)

        # Output box
        tk.Label(root, text="Prediction Output:").grid(row=4, column=0, sticky="nw", padx=10)
        self.output_box = tk.Text(root, height=10, width=80, state="disabled", bg="#f4f4f4", fg="black")
        self.output_box.grid(row=4, column=1, columnspan=2, padx=10, pady=10)

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Email files", "*.eml *.txt")])
        if filepath:
            self.email_path.set(filepath)
            self.processed_txt_path = None
            self.clear_email_viewer()

    def run_pipeline(self):
        file_path = self.email_path.get()
        model = self.model_choice.get()

        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid .eml or .txt file.")
            return

        # If .eml, convert to .txt
        if file_path.endswith(".eml"):
            txt_file = file_path.replace(".eml", ".txt")
            try:
                subprocess.run(["python", "process.py", file_path, txt_file], check=True)
                file_path = txt_file
                self.processed_txt_path = txt_file
            except subprocess.CalledProcessError as e:
                messagebox.showerror("Processing Error", f"Failed to process .eml file:\n{e.stderr or e}")
                return
        else:
            self.processed_txt_path = file_path

        self.load_email_content()

        # Predict using the model
        try:
            result = subprocess.run(
                ["python", "main.py", "predict", "--model", model, "--email", file_path],
                check=True, capture_output=True, text=True
            )
            output_lines = result.stdout.strip().splitlines()
            last_3 = "\n".join(output_lines[-3:])

            self.output_box.configure(state="normal")
            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, last_3)
            self.output_box.configure(state="disabled")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Prediction Error", f"Prediction failed:\n{e.stderr or e}")

    def load_email_content(self):
        if not self.processed_txt_path or not os.path.exists(self.processed_txt_path):
            return

        try:
            with open(self.processed_txt_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read email text:\n{str(e)}")
            return

        self.email_viewer.configure(state="normal")
        self.email_viewer.delete("1.0", tk.END)
        self.email_viewer.insert("1.0", content)
        self.email_viewer.configure(state="disabled")

    def clear_email_viewer(self):
        self.email_viewer.configure(state="normal")
        self.email_viewer.delete("1.0", tk.END)
        self.email_viewer.configure(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmailClassifierGUI(root)
    root.mainloop()