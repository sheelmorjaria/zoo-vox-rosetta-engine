from manim import *
import numpy as np

class FullPresentation(Scene):
    def construct(self):
        # ==========================================
        # EPISODE 1: The "Graded Continuum" Problem
        # ==========================================
        self.play(Write(Text("Episode 1: The Problem", font_size=24).to_edge(UP)))
        self.episode_1()
        self.wait(1)
        self.clear()
        
        # ==========================================
        # EPISODE 2: The 112D Stack
        # ==========================================
        self.play(Write(Text("Episode 2: The Features", font_size=24).to_edge(UP)))
        self.episode_2()
        self.wait(1)
        self.clear()
        
        # ==========================================
        # EPISODE 3: Neural Boundary Detection
        # ==========================================
        self.play(Write(Text("Episode 3: Segmentation", font_size=24).to_edge(UP)))
        self.episode_3()
        self.wait(1)
        self.clear()
        
        # ==========================================
        # EPISODE 4: Discovering Syntax
        # ==========================================
        self.play(Write(Text("Episode 4: Discovery", font_size=24).to_edge(UP)))
        self.episode_4()
        self.wait(2)

    # ---------------- EPISODE 1 ----------------
    def episode_1(self):
        # Title
        title = Text("The 'Graded Continuum' Problem", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Setup Axes
        axes_left = Axes(x_range=[0, 4], y_range=[0, 4], x_length=4, y_length=4).shift(LEFT * 3.5)
        axes_right = Axes(x_range=[0, 4], y_range=[0, 4], x_length=4, y_length=4).shift(RIGHT * 3.5)
        
        label_left = Text("Birds (Discrete)", font_size=24).next_to(axes_left, DOWN)
        label_right = Text("Bats (Graded)", font_size=24).next_to(axes_right, DOWN)

        self.play(Create(axes_left), Create(axes_right), Write(label_left), Write(label_right))

        # Left: Discrete Clusters
        dots_1 = VGroup(*[Dot(axes_left.c2p(np.random.normal(1, 0.2), np.random.normal(3, 0.2)), color=RED) for _ in range(20)])
        dots_2 = VGroup(*[Dot(axes_left.c2p(np.random.normal(3, 0.2), np.random.normal(1, 0.2)), color=BLUE) for _ in range(20)])
        self.play(Create(dots_1), Create(dots_2), run_time=2)
        
        # Right: Dense Cloud
        dots_cloud = VGroup(*[Dot(axes_right.c2p(np.random.normal(2, 0.8), np.random.normal(2, 0.8)), color=TEAL, radius=0.04) for _ in range(200)])
        self.play(Create(dots_cloud), run_time=3)
        
        conclusion = Text("Standard classifiers fail on the 'Cloud'", color=YELLOW, font_size=28).next_to(label_right, DOWN * 1.5)
        self.play(Write(conclusion))
        self.wait(2)

    # ---------------- EPISODE 2 ----------------
    def episode_2(self):
        title = Text("The 112D Micro-Dynamics Stack", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))

        # Use a VGroup to hold the stack for easy centering
        stack = VGroup()
        
        # Layer 1
        block1 = Rectangle(height=0.8, width=5, fill_color=RED, fill_opacity=0.7, stroke_color=WHITE)
        text1 = Text("Layer 1: Physics (46D)", font_size=20).move_to(block1)
        block1.add(text1)
        
        # Layer 2
        block2 = Rectangle(height=0.8, width=5, fill_color=BLUE, fill_opacity=0.7, stroke_color=WHITE)
        text2 = Text("Layer 2: Macro Texture (30D)", font_size=20).move_to(block2)
        block2.add(text2)
        
        # Layer 3
        block3 = Rectangle(height=0.8, width=5, fill_color=GREEN, fill_opacity=0.7, stroke_color=WHITE)
        text3 = Text("Layer 3: Micro Texture (36D)", font_size=20).move_to(block3)
        block3.add(text3)

        # Arrange vertically
        block2.next_to(block1, UP, buff=0.1)
        block3.next_to(block2, UP, buff=0.1)
        
        stack.add(block1, block2, block3)
        stack.move_to(ORIGIN) # Center the whole stack

        # Animate layers appearing one by one
        self.play(FadeIn(block1), run_time=0.5)
        self.wait(0.5)
        self.play(FadeIn(block2), run_time=0.5)
        self.wait(0.5)
        self.play(FadeIn(block3), run_time=0.5)
        self.wait(1)

        # Total Label
        total_label = Text("Total: 112 Dimensions", font_size=32, color=WHITE)
        total_label.next_to(stack, DOWN, buff=0.5)
        self.play(Write(total_label))
        self.wait(2)

    # ---------------- EPISODE 3 ----------------
    def episode_3(self):
        title = Text("Neural Boundary Detection (NBD)", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))

        axes = Axes(x_range=[0, 10, 1], y_range=[-1.5, 1.5, 0.5], x_length=10, y_length=3, axis_config={"include_tip": False})
        axes.shift(DOWN * 0.5)
        
        # Signal function
        def signal_func(x):
            if x < 5: return 0.8 * np.sin(2 * x)
            else: return 0.8 * np.sin(2 * x) + 0.3 * np.sin(20 * x)

        signal = axes.plot(signal_func, color=BLUE)
        signal_label = Text("Graded Vocalization", font_size=24).next_to(axes, UP)
        self.play(Create(axes), Create(signal), Write(signal_label))

        # --- Failure ---
        fail_text = Text("Energy-Based Detection (Fails)", color=RED, font_size=24).to_edge(LEFT).shift(UP)
        self.play(Write(fail_text))
        
        energy_line = DashedLine(axes.c2p(9.5, -1.5), axes.c2p(9.5, 1.5), color=RED)
        self.play(Create(energy_line))
        self.wait(0.5)
        
        miss_circle = Circle(color=RED, radius=0.5).move_to(axes.c2p(5, 0.5))
        miss_label = Text("Missed!", font_size=18, color=RED).next_to(miss_circle, RIGHT)
        self.play(Create(miss_circle), Write(miss_label))
        self.wait(1)
        
        # --- Success ---
        self.play(FadeOut(energy_line), FadeOut(miss_circle), FadeOut(miss_label), FadeOut(fail_text))
        
        success_text = Text("NBD Segmentation (Success)", color=GREEN, font_size=24).to_edge(LEFT).shift(UP)
        self.play(Write(success_text))
        
        nbd_line = DashedLine(axes.c2p(5, -1.5), axes.c2p(5, 1.5), color=GREEN)
        self.play(Create(nbd_line))
        
        seg1 = Text("Syllable A", font_size=18, color=GREEN).move_to(axes.c2p(2.5, 1.0))
        seg2 = Text("Syllable B", font_size=18, color=GREEN).move_to(axes.c2p(7.5, 1.0))
        self.play(Write(seg1), Write(seg2))
        self.wait(2)

    # ---------------- EPISODE 4 ----------------
    def episode_4(self):
        title = Text("Discovering Syntax in 'Noise'", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))

        # Timeline
        line = Line(LEFT * 6, RIGHT * 6, color=WHITE)
        line.shift(DOWN)
        self.play(Create(line))

        # Sequence Data
        syllables = [("336", RED), ("336", RED), ("391", GREEN), ("391", GREEN), ("391", GREEN)]
        start_pos = LEFT * 5
        step = 2.5
        
        dots = VGroup()
        labels = VGroup()
        
        # Animate dots appearing sequentially
        for i, (name, color) in enumerate(syllables):
            dot = Dot(point=start_pos + RIGHT * i * step, color=color, radius=0.2)
            label = Text(name, font_size=24).next_to(dot, UP)
            
            # Play animation for each dot immediately
            self.play(Create(dot), Write(label), run_time=0.4)
            dots.add(dot)
            labels.add(label)
            
        self.wait(0.5)

        # Highlight the pattern
        bracket1 = BraceBetweenPoints(dots[2].get_center(), dots[3].get_center(), direction=DOWN)
        bracket2 = BraceBetweenPoints(dots[3].get_center(), dots[4].get_center(), direction=DOWN)
        
        self.play(Create(bracket1), Create(bracket2))
        
        result_text = Text("Discrete Syntax Detected!", font_size=28, color=YELLOW).next_to(line, DOWN)
        context_text = Text("Context: Territorial (82% Purity)", font_size=24).next_to(result_text, DOWN)
        
        self.play(Write(result_text), Write(context_text))
        self.wait(3)
