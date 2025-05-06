import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import os

class EmailClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Email Classifier")

        self.email_path = tk.StringVar()
        self.model_choice = tk.StringVar(value="nb")

        # File input
        tk.Label(root, text="Select .eml or .txt file:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(root, textvariable=self.email_path, width=50).grid(row=0, column=1, padx=10)
        tk.Button(root, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)

        # Model selection
        tk.Label(root, text="Model:").grid(row=1, column=0, sticky="w", padx=10)
        ttk.Combobox(root, textvariable=self.model_choice, values=["nb", "svm", "baseline"], state="readonly", width=10).grid(row=1, column=1, sticky="w", padx=10)

        # Run button
        tk.Button(root, text="Run Prediction", command=self.run_pipeline).grid(row=2, column=1, pady=10)

        # Output box
        self.output_box = tk.Text(root, height=10, width=80, state="disabled", bg="#f4f4f4", fg="black")
        self.output_box.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Email files", "*.eml *.txt")])
        if filepath:
            self.email_path.set(filepath)

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
                file_path = txt_file  # use the converted file for prediction
            except subprocess.CalledProcessError as e:
                messagebox.showerror("Processing Error", f"Failed to process .eml file:\n{e.stderr or e}")
                return

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

if __name__ == "__main__":
    root = tk.Tk()
    app = EmailClassifierGUI(root)
    root.mainloop()