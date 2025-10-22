import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
from scipy import signal

plt.style.use("dark_background")

COLORS = {"primary": "#00F5FF", "secondary": "#FF0080", "accent": "#FFD700", "success": "#00FFA3",
          "wave1": "#A855F7", "wave2": "#FF0080", "envelope": "#FFD700", "bg_dark": "#0B0F1F",
          "bg_light": "#1E2538", "bg_panel": "#252D42", "text": "#F0F4FF", "text_dim": "#B8C5E0",
          "grid": "#2D3561", "border": "#3D4A6E"}

PRESETS = {"🎼 Musical": {"f1": 440, "f2": 445, "A": 1.0, "damping": 0.0},
           "🔊 Bass": {"f1": 120, "f2": 125, "A": 1.5, "damping": 0.2},
           "🎵 Treble": {"f1": 380, "f2": 390, "A": 1.0, "damping": 0.1},
           "⚡ Fast": {"f1": 300, "f2": 320, "A": 1.2, "damping": 0.0},
           "🌊 Slow": {"f1": 250, "f2": 252, "A": 1.0, "damping": 0.0},
           "💫 Damped": {"f1": 260, "f2": 250, "A": 1.0, "damping": 2.5}}

class BeatSimulatorPro:
    def __init__(self):
        self.A, self.f1, self.f2, self.damping = 1.0, 260, 250, 0.0
        self.phase1, self.phase2, self.sample_rate = 0, 0, 2000
        self.current_preset = None
        
        self.fig = plt.figure(figsize=(20, 11), facecolor=COLORS["bg_dark"])
        self.fig.canvas.manager.set_window_title("🎵 Ultimate Beat Simulator - Best UI Ever")
        
        gs = GridSpec(5, 3, figure=self.fig, hspace=0.4, wspace=0.35,
                     left=0.06, right=0.97, top=0.93, bottom=0.12)
        
        self.ax_main = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_spectrum = self.fig.add_subplot(gs[0, 2])
        self.ax_phase = self.fig.add_subplot(gs[1, 2])
        self.ax_individual = self.fig.add_subplot(gs[2, 0:2])
        self.ax_envelope = self.fig.add_subplot(gs[2, 2])
        self.ax_heatmap = self.fig.add_subplot(gs[3, 0:2])
        self.ax_info = self.fig.add_subplot(gs[3, 2])
        self.ax_info.axis("off")
        self.ax_presets = self.fig.add_subplot(gs[4, :])
        self.ax_presets.axis("off")
        
        for ax in [self.ax_main, self.ax_spectrum, self.ax_phase, self.ax_individual, self.ax_envelope, self.ax_heatmap]:
            ax.set_facecolor(COLORS["bg_light"])
            for spine in ax.spines.values():
                spine.set_color(COLORS["border"])
                spine.set_linewidth(2)
            ax.tick_params(colors=COLORS["text"], labelsize=9)
            ax.xaxis.label.set_color(COLORS["text"])
            ax.yaxis.label.set_color(COLORS["text"])
            ax.title.set_color(COLORS["text"])
            ax.grid(True, alpha=0.15, linestyle="--", color=COLORS["grid"], linewidth=0.8)
        
        self.create_controls()
        self.create_preset_buttons()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.update(None)
        
    def create_controls(self):
        sh, sw = 0.025, 0.22
        ax_f1 = plt.axes([0.12, 0.06, sw, sh], facecolor=COLORS["bg_panel"])
        self.slider_f1 = Slider(ax_f1, "Freq 1 (Hz)", 100, 500, valinit=self.f1, valstep=1, color=COLORS["wave1"])
        ax_f2 = plt.axes([0.12, 0.025, sw, sh], facecolor=COLORS["bg_panel"])
        self.slider_f2 = Slider(ax_f2, "Freq 2 (Hz)", 100, 500, valinit=self.f2, valstep=1, color=COLORS["wave2"])
        ax_amp = plt.axes([0.42, 0.06, sw, sh], facecolor=COLORS["bg_panel"])
        self.slider_amp = Slider(ax_amp, "Amplitude", 0.1, 2.0, valinit=self.A, valstep=0.1, color=COLORS["accent"])
        ax_damp = plt.axes([0.42, 0.025, sw, sh], facecolor=COLORS["bg_panel"])
        self.slider_damp = Slider(ax_damp, "Damping", 0.0, 5.0, valinit=self.damping, valstep=0.1, color=COLORS["success"])
        
        for slider in [self.slider_f1, self.slider_f2, self.slider_amp, self.slider_damp]:
            slider.on_changed(self.update)
        
    def create_preset_buttons(self):
        self.preset_buttons = []
        bw, bh, sx, sp, yp = 0.14, 0.04, 0.08, 0.015, 0.005
        for i, (name, params) in enumerate(PRESETS.items()):
            ax_btn = plt.axes([sx + i * (bw + sp), yp, bw, bh])
            btn = Button(ax_btn, name, color=COLORS["bg_panel"], hovercolor=COLORS["primary"])
            btn.label.set_color(COLORS["text"])
            btn.label.set_fontsize(9)
            btn.label.set_fontweight("bold")
            btn.on_clicked(lambda e, p=params, n=name: self.apply_preset(p, n))
            self.preset_buttons.append(btn)
    
    def apply_preset(self, params, name):
        self.current_preset = name
        self.slider_f1.set_val(params["f1"])
        self.slider_f2.set_val(params["f2"])
        self.slider_amp.set_val(params["A"])
        self.slider_damp.set_val(params["damping"])
        print(f"✨ Applied: {name}")
        
    def on_key_press(self, event):
        if event.key == "r":
            for s, v in zip([self.slider_f1, self.slider_f2, self.slider_amp, self.slider_damp], [260, 250, 1.0, 0.0]):
                s.set_val(v)
            print("🔄 Reset")
        elif event.key == "s":
            self.fig.savefig("beat_viz.png", dpi=300, facecolor=COLORS["bg_dark"])
            print("💾 Saved beat_viz.png")
        elif event.key == "h":
            print("\n⌨️  Shortcuts: R-Reset | S-Save | H-Help | Q-Quit\n")
        elif event.key == "q":
            plt.close()
        
    def calculate_waves(self):
        self.f1, self.f2 = self.slider_f1.val, self.slider_f2.val
        self.A, self.damping = self.slider_amp.val, self.slider_damp.val
        
        beat_freq = abs(self.f1 - self.f2) or 0.1
        beat_period, avg_freq = 1 / beat_freq, (self.f1 + self.f2) / 2
        
        t_end = max(2, min(5, int(2.0 * beat_freq))) * beat_period
        self.t = np.linspace(0, t_end, self.sample_rate)
        
        self.y1 = self.A * np.sin(2 * np.pi * self.f1 * self.t)
        self.y2 = self.A * np.sin(2 * np.pi * self.f2 * self.t)
        
        if self.damping > 0:
            damp_env = np.exp(-self.damping * self.t)
            self.y1, self.y2 = self.y1 * damp_env, self.y2 * damp_env
        
        self.y_resultant = self.y1 + self.y2
        self.envelope_upper = 2 * self.A * np.abs(np.cos(2 * np.pi * beat_freq / 2 * self.t))
        if self.damping > 0:
            self.envelope_upper *= np.exp(-self.damping * self.t)
        self.envelope_lower = -self.envelope_upper
        
        return beat_freq, beat_period, avg_freq
    
    def plot_main_waveform(self, beat_freq):
        self.ax_main.clear()
        self.ax_main.fill_between(self.t, self.envelope_lower, self.envelope_upper,
                                  alpha=0.2, color=COLORS["envelope"], zorder=1)
        self.ax_main.plot(self.t, self.envelope_upper, "--", color=COLORS["envelope"], linewidth=2.5, alpha=0.9, zorder=2)
        self.ax_main.plot(self.t, self.envelope_lower, "--", color=COLORS["envelope"], linewidth=2.5, alpha=0.9, zorder=2)
        
        points = np.array([self.t, self.y_resultant]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(-2*self.A, 2*self.A)
        lc = plt.matplotlib.collections.LineCollection(segments, cmap="twilight_shifted", norm=norm, linewidth=3.5, zorder=3)
        lc.set_array(self.y_resultant)
        self.ax_main.add_collection(lc)
        
        self.ax_main.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        self.ax_main.set_ylabel("Amplitude", fontsize=12, fontweight="bold")
        title = f"🎵 Beat Wave | Freq: {beat_freq:.2f} Hz"
        if self.current_preset:
            title += f" | {self.current_preset}"
        self.ax_main.set_title(title, fontsize=14, fontweight="bold", pad=15)
        self.ax_main.set_xlim(self.t[0], self.t[-1])
        self.ax_main.set_ylim(-2.5*self.A, 2.5*self.A)
        self.ax_main.axhline(y=0, color=COLORS["text_dim"], linewidth=1, alpha=0.4, linestyle=":")
        
    def plot_spectrum(self):
        self.ax_spectrum.clear()
        fft_vals = np.fft.fft(self.y_resultant)
        fft_freq = np.fft.fftfreq(len(self.y_resultant), 1/self.sample_rate)
        pos_mask = fft_freq > 0
        fft_freq, fft_mag = fft_freq[pos_mask], np.abs(fft_vals[pos_mask]) / len(self.y_resultant)
        
        self.ax_spectrum.bar(fft_freq, fft_mag, width=2, color=COLORS["primary"], alpha=0.8, edgecolor=COLORS["accent"], linewidth=1.5)
        self.ax_spectrum.set_xlabel("Frequency (Hz)", fontsize=10, fontweight="bold")
        self.ax_spectrum.set_ylabel("Magnitude", fontsize=10, fontweight="bold")
        self.ax_spectrum.set_title("📊 Spectrum (FFT)", fontsize=12, fontweight="bold", pad=10)
        self.ax_spectrum.set_xlim(0, max(self.f1, self.f2) + 100)
        
    def plot_phase_space(self):
        self.ax_phase.clear()
        points = np.array([self.y1, self.y2]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, len(self.t))
        lc = plt.matplotlib.collections.LineCollection(segments, cmap="viridis", norm=norm, linewidth=2.5, alpha=0.9)
        lc.set_array(np.arange(len(self.t)))
        self.ax_phase.add_collection(lc)
        
        self.ax_phase.set_xlabel("Wave 1", fontsize=10, fontweight="bold")
        self.ax_phase.set_ylabel("Wave 2", fontsize=10, fontweight="bold")
        self.ax_phase.set_title("🌀 Phase Space", fontsize=12, fontweight="bold", pad=10)
        self.ax_phase.set_xlim(-1.5*self.A, 1.5*self.A)
        self.ax_phase.set_ylim(-1.5*self.A, 1.5*self.A)
        self.ax_phase.set_aspect("equal")
        
    def plot_individual_waves(self):
        self.ax_individual.clear()
        self.ax_individual.plot(self.t, self.y1, color=COLORS["wave1"], linewidth=2.5, alpha=0.8, label=f"Wave 1: {self.f1:.0f} Hz")
        self.ax_individual.plot(self.t, self.y2, color=COLORS["wave2"], linewidth=2.5, alpha=0.8, label=f"Wave 2: {self.f2:.0f} Hz")
        self.ax_individual.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        self.ax_individual.set_ylabel("Amplitude", fontsize=10, fontweight="bold")
        self.ax_individual.set_title("🎼 Individual Waves", fontsize=12, fontweight="bold", pad=10)
        self.ax_individual.legend(loc="upper right", framealpha=0.95, fontsize=10)
        self.ax_individual.set_xlim(self.t[0], self.t[-1])
        self.ax_individual.axhline(y=0, color=COLORS["text_dim"], linewidth=1, alpha=0.4, linestyle=":")
        
    def plot_envelope_analysis(self, beat_freq):
        self.ax_envelope.clear()
        envelope = np.abs(signal.hilbert(self.y_resultant))
        self.ax_envelope.fill_between(self.t, 0, envelope, color=COLORS["success"], alpha=0.7)
        self.ax_envelope.plot(self.t, envelope, color=COLORS["success"], linewidth=2.5)
        self.ax_envelope.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        self.ax_envelope.set_ylabel("Intensity", fontsize=10, fontweight="bold")
        self.ax_envelope.set_title("💫 Beat Intensity", fontsize=12, fontweight="bold", pad=10)
        self.ax_envelope.set_xlim(self.t[0], self.t[-1])
        
    def plot_heatmap(self):
        self.ax_heatmap.clear()
        f, t_spec, Sxx = signal.spectrogram(self.y_resultant, fs=self.sample_rate, nperseg=256, noverlap=200)
        im = self.ax_heatmap.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="magma", vmin=-40, vmax=20)
        self.ax_heatmap.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        self.ax_heatmap.set_ylabel("Frequency (Hz)", fontsize=10, fontweight="bold")
        self.ax_heatmap.set_title("🔥 Spectrogram", fontsize=12, fontweight="bold", pad=10)
        self.ax_heatmap.set_ylim(0, max(self.f1, self.f2) + 100)
        cbar = plt.colorbar(im, ax=self.ax_heatmap, pad=0.02)
        cbar.set_label("Power (dB)", fontsize=9, color=COLORS["text"])
        cbar.ax.tick_params(colors=COLORS["text"], labelsize=8)
        
    def plot_info_panel(self, beat_freq, beat_period, avg_freq):
        self.ax_info.clear()
        self.ax_info.axis("off")
        info = f"""
╔═══════════════════════════════╗
║  SIMULATION PARAMETERS      ║
╚═══════════════════════════════╝

📐 Wave Properties:
  - Freq 1: {self.f1:.1f} Hz
  - Freq 2: {self.f2:.1f} Hz
  - Amplitude: {self.A:.2f}
  - Damping: {self.damping:.2f}

🎵 Beat Analysis:
  - Beat Freq: {beat_freq:.3f} Hz
  - Period: {beat_period:.4f} s
  - Avg Freq: {avg_freq:.1f} Hz
  - Δf: {abs(self.f1-self.f2):.1f} Hz

⚡ Physics:
  - Superposition: ✓
  - Interference: ✓
  - Damping: {"✓" if self.damping > 0 else "✗"}

⌨️  Shortcuts:
  R-Reset | S-Save
  H-Help  | Q-Quit
"""
        self.ax_info.text(0.05, 0.95, info, transform=self.ax_info.transAxes, fontsize=9, verticalalignment="top",
                         fontfamily="monospace", color=COLORS["text"],
                         bbox=dict(boxstyle="round,pad=1.2", facecolor=COLORS["bg_panel"], 
                                  edgecolor=COLORS["primary"], linewidth=2.5, alpha=0.95))
        
    def update(self, val):
        beat_freq, beat_period, avg_freq = self.calculate_waves()
        self.plot_main_waveform(beat_freq)
        self.plot_spectrum()
        self.plot_phase_space()
        self.plot_individual_waves()
        self.plot_envelope_analysis(beat_freq)
        self.plot_heatmap()
        self.plot_info_panel(beat_freq, beat_period, avg_freq)
        self.fig.canvas.draw_idle()
        
    def show(self):
        plt.show()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎵 ULTIMATE BEAT SIMULATOR - BEST UI EVER")
    print("="*70)
    print("\n✨ Features: 6 Presets | FFT | Lissajous | Spectrogram | Keyboard Shortcuts")
    print("⌨️  Shortcuts: R-Reset | S-Save (300 DPI) | H-Help | Q-Quit")
    print("="*70 + "\n")
    
    simulator = BeatSimulatorPro()
    simulator.show()
