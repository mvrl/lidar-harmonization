from manim import *
import numpy as np
from pathlib import Path
import code

# Animation story board:
# 1. Show two planes flying across the scene (perpendicular, sequential?).
#    Differently colored bounding boxes appear behind the planes to indicate the
#    LiDAR that is being collected on the ground. 
#
# 2. The LiDAR rectangles are replaced by evenly spaced dots of the same color.
#    Initially, the dots occupy the same grid spots (in the overlap). Perhaps
#    it makes sense to show this and how it is simple to approximate a function
#    from this. 
#
# 3. The dots from the second plane's LiDAR are shifted (there are several ways
#    to do this: a linear shift, jitter,...). We show how it is no longer simple
#    to build a transformation. 

# 4. Explanation of our method... animate that process (the rest of the owl)

# 5. Change the color of the second plane's LiDAR dots to the color of the 
#    first plane's dots.

class OpenAnim(Scene):
    def construct(self):
        grid = NumberPlane()
        # Create the scene
        self.add(grid)
        self.remove(grid)  # this is hacky

        # Scene 1:
        """
        LiDAR has become a very powerful tool for surveying large regions very
        quickly. LiDAR sensors can be affixed to planes, and then these planes
        can very easiliy sweep large regions by moving back and forth across the
        area of interest. This method is powerful, as LiDAR is extremely accurate
        ....

        """
        plane1_pos1 = np.array([-8, 3, 0])
        plane1_pos2 = np.array([8, 3, 0])
        plane1 = ImageMobject("plane.png").move_to(plane1_pos1)# .rotate(-PI/4)
        plane1.height = 1
        plane1.width = 1
        plane1.rotate(-PI/2)
        
        for i in range(4):

            rect1_1 = Rectangle(color=RED, height=2, width=1).move_to(plane1_pos1)
            rect1_1.set_fill(RED, opacity=0.5)
            rect1_2 = Rectangle(color=RED, height=2, width=32).move_to(plane1_pos1)
            rect1_2.set_fill(RED, opacity=0.5)
            self.add(rect1_1)
            self.add(plane1)
            
            self.play(
                ApplyMethod(plane1.move_to, plane1_pos2),
                Transform(rect1_1, rect1_2), 
                rate_func=linear, run_time=3)
            self.wait()

            self.remove(plane1)
            plane1_pos1[1] -= 2
            plane1_pos2[1] -= 2
            plane1_pos1[0] = -plane1_pos1[0]
            plane1_pos2[0] = -plane1_pos2[0]
            plane1.move_to(plane1_pos1)
            plane1.rotate(PI)

        self.wait(2)
        self.clear()
        self.wait(2)

class OverlapAnim(Scene):
    def construct(self):
        grid = NumberPlane()
        # Create the scene
        self.add(grid)
        self.remove(grid)  # this is hacky

        # Scene 2: 
        """
        Sometimes, data acquisition takes multiple flights, either with multiple 
        planes or sequential flights over a longer period of time. Conditions 
        such as humidity, temperature, and wetness all affect the intensity 
        measurement. It may even be the case that sensors calibration is just a 
        little bit different. This presents a problem when trying to compare 
        scans that are ajacent or overlapping: since the intensity is different,
        our ability to use intensity as a feature for tasks such as 
        classification or detection is impaired.
        """
        plane1_pos1 = np.array([-8, 0, 0])
        plane1_pos2 = np.array([8, 0, 0])
        plane1 = ImageMobject("plane.png").move_to(plane1_pos1)# .rotate(-PI/4)
        plane1.height = 1
        plane1.width = 1
        plane1.rotate(-PI/2)

        rect1_1 = Rectangle(color=RED, height=2, width=1).move_to(plane1_pos1)
        rect1_1.set_fill(RED, opacity=0.5)
        rect1_2 = Rectangle(color=RED, height=2, width=32).move_to(plane1_pos1)
        rect1_2.set_fill(RED, opacity=0.5)

        plane2_pos1 = np.array([0, -5, 0])
        plane2_pos2 = np.array([0, 5, 0])
        plane2 = ImageMobject("plane.png").move_to(plane2_pos1)# .rotate(-3*PI/4)
        plane2.height = 1
        plane2.width = 1
        rect2_1 = Rectangle(color=BLUE, height=1, width=2).move_to(plane2_pos1)
        rect2_1.set_fill(BLUE)
        rect2_2 = Rectangle(color=BLUE, height=20, width=2).move_to(plane2_pos1)
        rect2_2.set_fill(BLUE, opacity=0.5)

        self.add(rect1_1)
        self.add(rect2_1)
        self.add(plane2)
        self.add(plane1)
        
        self.wait()
        self.play(
            ApplyMethod(plane1.move_to, plane1_pos2),
            Transform(rect1_1, rect1_2),
            rate_func=linear, run_time=3),
        self.wait()

        self.play(
            ApplyMethod(plane2.move_to, plane2_pos2),
            Transform(rect2_1, rect2_2),
            rate_func=linear, run_time=3)

        self.wait(2)
        self.clear()
        self.wait(2)

class OverlapAnimDots(Scene):
    def construct(self):
        grid = NumberPlane()
        self.add(grid)
        self.remove(grid)

        # Scene 3:
        """
        Let's take a closer look at this. The point cloud data is now shown.
        In an ideal world, we would have points that overlap nicely. In this
        case, we could just approximate a function using our favorite function
        approximation algorithm. However, it is rare for there to be a case like
        this. 
        """
        density=.5  # (np defaults to 50)
        dot_radius = 0.12  # (manim defaults to 0.08)
        vertical_space_ratio = 8

        plane1_pos1 = np.array([-8, 0, 0])
        rect1 = Rectangle(color=RED, height=2, width=16).set_fill(RED, opacity=0.5)
        rect2 = Rectangle(color=BLUE, height=10, width=2).set_fill(BLUE, opacity=0.5)
        self.add(rect1)
        self.add(rect2)

        dots1_coords = []
        for i in np.arange(-8, 8+density, density):
            for j in np.arange(-1, 1+density, density):
                dots1_coords.append(np.array([i, j, 0]))

        dots1 = [Dot(i, radius=dot_radius).set_fill(RED, opacity=1) for i in dots1_coords]


        dots2_coords = []
        for i in np.arange(-1, 1+density, density):
            for j in np.arange(-5, 5+density, density):
                dots2_coords.append(np.array([i, j, 0]))

        dots2 = [Dot(i, radius=dot_radius).set_fill(BLUE, opacity=1) for i in dots2_coords]

        dot1set = set([tuple(x) for x in dots1_coords])
        dot2set = set([tuple(x) for x in dots2_coords])
        dots_overlap_coords = np.array([x for x in dot1set & dot2set])

        dots1_overlap = [Dot(i, radius=dot_radius).set_fill(RED, opacity=1) for i in dots_overlap_coords]
        dots2_overlap = [Dot(i, radius=dot_radius).set_fill(BLUE, opacity=1) for i in dots_overlap_coords]

        # transform overlap points into a straight vertical line
        num_dots = len(dots1_overlap)

        dots_vertical_coords = []
        for i in np.linspace(-num_dots/vertical_space_ratio, num_dots/vertical_space_ratio, num_dots):
            dots_vertical_coords.append([0, i, 0])

        dots1_vertical = [Dot(i, radius=dot_radius/2).set_fill(RED, opacity=1) for i in dots_vertical_coords]
        dots2_vertical = [Dot(i, radius=dot_radius/2).set_fill(BLUE, opacity=1) for i in dots_vertical_coords]

        dots_overlap_to_vertical_anim = []
        for i in range(num_dots):
            dots_overlap_to_vertical_anim.append(ReplacementTransform(dots1_overlap[i], dots1_vertical[i]))
            dots_overlap_to_vertical_anim.append(ReplacementTransform(dots2_overlap[i], dots2_vertical[i]))

        # Reset overlap with harmonized colors
        dots1_overlap_h = [Dot(i, radius=dot_radius).set_fill(RED, opacity=1) for i in dots_overlap_coords]
        dots2_overlap_h = [Dot(i, radius=dot_radius).set_fill(RED, opacity=1) for i in dots_overlap_coords]

        dots_vertical_to_overlap_anim = []
        for i in range(num_dots):
            dots_vertical_to_overlap_anim.append(ReplacementTransform(dots1_vertical[i], dots1_overlap_h[i]))
            dots_vertical_to_overlap_anim.append(ReplacementTransform(dots2_vertical[i], dots2_overlap_h[i]))

        dots1_anim = AnimationGroup(*[GrowFromCenter(d) for d in dots1])
        dots2_anim = AnimationGroup(*[GrowFromCenter(d) for d in dots2])
        dots1_group = VGroup(*dots1)
        dots2_group = VGroup(*dots2)
        dots1_overlap_group = VGroup(*dots1_overlap)
        dots2_overlap_group = VGroup(*dots2_overlap)

        dots1_overlap_h_group = VGroup(*dots1_overlap_h)
        dots2_overlap_h_group = VGroup(*dots2_overlap_h)

        dots1_vertical_group = VGroup(*dots1_vertical)
        dots2_vertical_group = VGroup(*dots2_vertical)

        vertical_expression_tex = MathTex(r'I_x = f(H_x)')
        vertical_expression_tex_2 = MathTex(r'H_x = g(I_x), g = f^{-1}')

        self.play(
            dots1_anim,
            dots2_anim,
            FadeOut(rect1),
            FadeOut(rect2))
        self.wait()
        self.play(
            FadeIn(dots1_overlap_group),
            FadeIn(dots2_overlap_group),
            FadeOut(dots1_group),
            FadeOut(dots2_group)
            )
        self.wait()

        # show the two sets overlap
        self.play(
            dots1_overlap_group.animate.shift(np.array([3.0, 0, 0])),
            dots2_overlap_group.animate.shift(np.array([-3.0, 0, 0])))
        self.wait()

        # put them back
        self.play(
            dots1_overlap_group.animate.shift(np.array([-3.0, 0, 0])),
            dots2_overlap_group.animate.shift(np.array([3.0, 0, 0])))
        self.wait()

        # lay them out vertically
        self.play(AnimationGroup(*dots_overlap_to_vertical_anim))
        self.wait()

        # shift the dot sets
        self.play(
            dots1_vertical_group.animate.shift(np.array([3.0, 0, 0])),
            dots2_vertical_group.animate.shift(np.array([-3.0, 0, 0]))
            )
        self.wait()

        # write out our equation for kicks
        self.play(Write(vertical_expression_tex))
        self.wait()

        self.play(vertical_expression_tex.animate.shift(np.array([0, 1, 0])))
        self.wait()
        self.play(Write(vertical_expression_tex_2))
        self.wait()

        # harmonize dots
        self.play(dots2_vertical_group.animate.set_fill(RED, opacity=1))
        self.wait()

        # remove tex
        self.play(
            vertical_expression_tex.animate.shift(np.array([0, 10, 0])),
            vertical_expression_tex_2.animate.shift(np.array([0, 10, 0])))
        self.wait()
        self.remove(vertical_expression_tex)
        self.remove(vertical_expression_tex_2)

        # reverse the animation sequence
        # shift the dot sets
        self.play(
            dots1_vertical_group.animate.shift(np.array([-3.0, 0, 0])),
            dots2_vertical_group.animate.shift(np.array([3.0, 0, 0]))
            )
        self.wait()

        # put them back in the grid
        self.play(AnimationGroup(*dots_vertical_to_overlap_anim))
        self.wait()

        # show the whole set
        self.play(
            FadeIn(dots2_group),
            FadeIn(dots1_group),
            FadeOut(dots1_overlap_h_group),
            FadeOut(dots2_overlap_h_group)
            )
        self.wait()

        self.play(dots2_group.animate.set_color(RED))
        self.wait()

        rect2.set_color(RED)

        self.play(
            FadeOut(dots2_group),
            FadeOut(dots1_group),
            FadeIn(rect1),
            FadeIn(rect2),
            rect1.animate.set_fill(RED_C, opacity=1),
            rect2.animate.set_fill(RED_C, opacity=1)
            )
        self.wait()


        self.play(
            FadeOut(rect1),
            FadeOut(rect2))
        self.wait()
        


