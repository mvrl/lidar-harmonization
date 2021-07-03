from manim import *
import numpy as np
from pathlib import Path
import code

MY_RED = RED
MY_PURPLE = PURPLE
MY_GREEN = GREEN
MY_BLUE = BLUE
MY_YELLOW = "#b9b946"
config.background_color = "#303030"  # google slides "simple dark" background color according to colorzilla

class AAA_TitleScreen(Scene):
    def construct(self):
        title_screen = ImageMobject("TitleScreen.png")
        self.add(title_screen)

        self.wait(3)
        self.play(
            FadeOut(title_screen),
            rate_func=linear, run_time=1)
        self.wait()

class A_OpenAnim(MovingCameraScene):
    def construct(self):
        grid = NumberPlane()
        # Create the scene
        self.add(grid)
        self.remove(grid)  # this is hacky, but makes positioning easier

        
        # a segment that makes it clear that the red zone is a point cloud might 
        # be useful
        plane1_pos1 = np.array([-8, 3, 0])
        plane1_pos2 = np.array([8, 3, 0])
        plane1 = ImageMobject("plane.png").move_to(plane1_pos1)# .rotate(-PI/4)
        plane1.height = 1
        plane1.width = 1
        plane1.rotate(-PI/2)
        positions = []
        for i in range(3):

            rect1_1 = Rectangle(color=MY_RED, height=2, width=1).move_to(plane1_pos1)
            rect1_1.set_fill(MY_RED, opacity=0.5)
            rect1_2 = Rectangle(color=MY_RED, height=2, width=32).move_to(plane1_pos1)
            rect1_2.set_fill(MY_RED, opacity=0.5)
            positions.append(plane1_pos1.copy())
            self.add(rect1_1)
            self.add(plane1)
            
            self.play(
                ApplyMethod(plane1.move_to, plane1_pos2),
                Transform(rect1_1, rect1_2), 
                rate_func=linear, run_time=5)

            self.remove(plane1)
            plane1_pos1[1] -= 2
            plane1_pos2[1] -= 2
            plane1_pos1[0] = -plane1_pos1[0]
            plane1_pos2[0] = -plane1_pos2[0]
            plane1.move_to(plane1_pos1)
            plane1.rotate(PI)

        # replace the weird rectangles with even rectangles:
        self.clear()
        for pos in positions:
            print(pos)
            rect = Rectangle(color=MY_RED, height=2, width=16).move_to([0, pos[1], 0])
            rect.set_fill(MY_RED, opacity=0.5)
            self.add(rect)

        self.wait()

        # plane1.rotate(PI)
        # self.add(plane1.move_to(np.array([-16, -3, 0])))
        plane1_pos2 = np.array([-25, -3, 0])
        rect2_1 = Rectangle(color=MY_RED, height=2, width=1).move_to(plane1_pos1)
        rect2_1.set_fill(MY_RED, opacity=0.5)
        self.add(rect2_1)
        rect2_2 = Rectangle(color=MY_RED, height=2, width=64).move_to(plane1_pos1)
        rect2_2.set_fill(MY_RED, opacity=0.5)
        self.play(
            self.camera.frame.animate.move_to(np.array([-16, -3, 0])),
            Transform(rect2_1, rect2_2),
            plane1.animate.move_to(plane1_pos2),
            rate_func=linear, run_time=12
            )
        self.wait()

        plane2_pos1 = np.array([-16, -8, 0])
        plane2_pos2 = np.array([-16, 2, 0])
        plane2 = ImageMobject("plane.png").move_to(plane2_pos1)# .rotate(-3*PI/4)
        plane2.height = 1
        plane2.width = 1
        rect2_1 = Rectangle(color=MY_BLUE, height=1, width=2).move_to(plane2_pos1)
        rect2_1.set_fill(MY_BLUE)
        rect2_2 = Rectangle(color=MY_BLUE, height=20, width=2).move_to(plane2_pos1)
        rect2_2.set_fill(MY_BLUE, opacity=0.5)
        self.add(rect2_1, plane2)

        self.play(
            ApplyMethod(plane2.move_to, plane2_pos2),
            Transform(rect2_1, rect2_2),
            rate_func=linear, run_time=4)

        self.wait()

        # self.wait(2)
        # self.clear()
        # self.wait(2)


class B_OverlapAnimDots(Scene):
    def construct(self):
        grid = NumberPlane()
        self.add(grid)
        self.remove(grid)


        density=.5  # (np defaults to 50)
        dot_radius = 0.12  # (manim defaults to 0.08)
        vertical_space_ratio = 8

        plane1_pos1 = np.array([-8, 0, 0])
        rect1 = Rectangle(color=MY_RED, height=2, width=16).set_fill(MY_RED, opacity=0.5)
        rect2 = Rectangle(color=MY_BLUE, height=10, width=2).set_fill(MY_BLUE, opacity=0.5)
        self.add(rect1)
        self.add(rect2)

        dots1_coords = []
        for i in np.arange(-8, 8+density, density):
            for j in np.arange(-1, 1+density, density):
                dots1_coords.append(np.array([i, j, 0]))

        dots1 = [Dot(i, radius=dot_radius).set_fill(MY_RED, opacity=1) for i in dots1_coords]


        dots2_coords = []
        for i in np.arange(-1, 1+density, density):
            for j in np.arange(-5, 5+density, density):
                dots2_coords.append(np.array([i, j, 0]))

        dots2 = [Dot(i, radius=dot_radius).set_fill(MY_BLUE, opacity=1) for i in dots2_coords]

        dot1set = set([tuple(x) for x in dots1_coords])
        dot2set = set([tuple(x) for x in dots2_coords])
        dots_overlap_coords = np.array([x for x in dot1set & dot2set])

        dots1_overlap = [Dot(i, radius=dot_radius).set_fill(MY_RED, opacity=1) for i in dots_overlap_coords]
        dots2_overlap = [Dot(i, radius=dot_radius).set_fill(MY_BLUE, opacity=1) for i in dots_overlap_coords]

        # transform overlap points into a straight vertical line
        num_dots = len(dots1_overlap)

        dots_vertical_coords = []
        for i in np.linspace(-num_dots/vertical_space_ratio, num_dots/vertical_space_ratio, num_dots):
            dots_vertical_coords.append([0, i, 0])

        dots1_vertical = [Dot(i, radius=dot_radius/2).set_fill(MY_RED, opacity=1) for i in dots_vertical_coords]
        dots2_vertical = [Dot(i, radius=dot_radius/2).set_fill(MY_BLUE, opacity=1) for i in dots_vertical_coords]

        dots_overlap_to_vertical_anim = []
        for i in range(num_dots):
            dots_overlap_to_vertical_anim.append(ReplacementTransform(dots1_overlap[i], dots1_vertical[i]))
            dots_overlap_to_vertical_anim.append(ReplacementTransform(dots2_overlap[i], dots2_vertical[i]))

        # Reset overlap with harmonized colors
        dots1_overlap_h = [Dot(i, radius=dot_radius).set_fill(MY_RED, opacity=1) for i in dots_overlap_coords]
        dots2_overlap_h = [Dot(i, radius=dot_radius).set_fill(MY_RED, opacity=1) for i in dots_overlap_coords]

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
        self.play(dots2_vertical_group.animate.set_fill(MY_RED, opacity=1))
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

        self.play(dots2_group.animate.set_color(MY_RED))
        self.wait( )

        rect2.set_color(MY_RED)

        self.play(
            FadeOut(dots2_group),
            FadeOut(dots1_group),
            FadeIn(rect1),
            FadeIn(rect2),
            rect1.animate.set_fill(MY_RED, opacity=1),
            rect2.animate.set_fill(MY_RED, opacity=1)
            )
        self.wait()

        # self.play(
        #     FadeOut(rect1),
        #     FadeOut(rect2))
        # self.wait()

class C_OverlapAnimMorePlanes(Scene):
    def construct(self):
        grid = NumberPlane()
        self.add(grid)
        self.remove(grid)

        rect1 = Rectangle(color=MY_RED, height=2, width=16).set_fill(MY_RED, opacity=1)
        rect2 = Rectangle(color=MY_RED, height=10, width=2).set_fill(MY_RED, opacity=1)
        self.add(rect1)
        self.add(rect2)

        # more planes fly over with different colors... they get harmonized too
        plane1_pos1 = np.array([-2, -5, 0])
        plane1_pos2 = np.array([-2, 5, 0])
        plane1 = ImageMobject("plane.png").move_to(plane1_pos1)
        plane1.height = 1
        plane1.width = 1
        # plane1.rotate(-PI)
        rect1_1 = Rectangle(color=MY_GREEN, height=1, width=2).move_to(plane1_pos1)
        rect1_1.set_fill(MY_GREEN, opacity=1)
        rect1_2 = Rectangle(color=MY_GREEN, height=20, width=2).move_to(plane1_pos1)
        rect1_2.set_fill(MY_GREEN, opacity=1)

        plane2_pos1 = np.array([2, 5, 0])
        plane2_pos2 = np.array([2, -5, 0])
        plane2 = ImageMobject("plane.png").move_to(plane2_pos1)
        plane2.height = 1
        plane2.width = 1
        plane2.rotate(-PI)
        rect2_1 = Rectangle(color=MY_PURPLE, height=1, width=2).move_to(plane2_pos1)
        rect2_1.set_fill(MY_PURPLE, opacity=1)
        rect2_2 = Rectangle(color=MY_PURPLE, height=20, width=2).move_to(plane2_pos1)
        rect2_2.set_fill(MY_PURPLE, opacity=1)

        rect1_2_red = Rectangle(color=MY_RED, height=20, width=2).move_to(plane1_pos1)
        rect1_2_red.set_fill(MY_RED, opacity=1)
        rect2_2_red = Rectangle(color=MY_RED, height=20, width=2).move_to(plane2_pos1)
        rect2_2_red.set_fill(MY_RED, opacity=1)

        ####
        plane3_pos1 = np.array([-4, 5, 0])
        plane3_pos2 = np.array([-4, -5, 0])
        plane3 = ImageMobject("plane.png").move_to(plane3_pos1)
        plane3.height = 1
        plane3.width = 1
        plane3.rotate(-PI)
        rect3_1 = Rectangle(color=MY_YELLOW, height=1, width=2).move_to(plane3_pos1)
        rect3_1.set_fill(MY_YELLOW, opacity=1)
        rect3_2 = Rectangle(color=MY_YELLOW, height=20, width=2).move_to(plane3_pos1)
        rect3_2.set_fill(MY_YELLOW, opacity=1)

        plane4_pos1 = np.array([4, -5, 0])
        plane4_pos2 = np.array([4, 5, 0])
        plane4 = ImageMobject("plane.png").move_to(plane4_pos1)
        plane4.height = 1
        plane4.width = 1
        # plane4.rotate(-PI)
        rect4_1 = Rectangle(color=MY_BLUE, height=1, width=2).move_to(plane4_pos1)
        rect4_1.set_fill(MY_BLUE, opacity=1)
        rect4_2 = Rectangle(color=MY_BLUE, height=20, width=2).move_to(plane4_pos1)
        rect4_2.set_fill(MY_BLUE, opacity=1)

        rect3_2_red = Rectangle(color=MY_RED, height=20, width=2).move_to(plane3_pos1)
        rect3_2_red.set_fill(MY_RED, opacity=1)
        rect4_2_red = Rectangle(color=MY_RED, height=20, width=2).move_to(plane4_pos1)
        rect4_2_red.set_fill(MY_RED, opacity=1)

        self.add(rect1_1)
        self.add(rect2_1)
        self.add(rect3_1)
        self.add(rect4_1)
        self.add(plane4)
        self.add(plane3)
        self.add(plane2)
        self.add(plane1)

        self.play(
            ApplyMethod(plane1.move_to, plane1_pos2),
            ApplyMethod(plane2.move_to, plane2_pos2),
            ReplacementTransform(rect1_1, rect1_2),
            ReplacementTransform(rect2_1, rect2_2),
            plane3.animate.move_to(plane3_pos2),
            plane4.animate.move_to(plane4_pos2),
            ReplacementTransform(rect3_1, rect3_2),
            ReplacementTransform(rect4_1, rect4_2),
            rate_func=linear, run_time=3)
        self.remove(rect1_1)
        self.remove(rect2_1)

        self.play(
            Transform(rect1_2, rect1_2_red),
            Transform(rect2_2, rect2_2_red),
            Transform(rect3_2, rect3_2_red),
            Transform(rect4_2, rect4_2_red))
        self.wait()

        self.play(
            FadeOut(rect1),
            FadeOut(rect2),
            FadeOut(rect1_2),
            FadeOut(rect2_2),
            FadeOut(rect3_2),
            FadeOut(rect4_2)
        )
        self.wait()


class D_OverlapAnimDotsActual(MovingCameraScene):
    def setup(self):
        MovingCameraScene.setup(self)

    def construct(self):
        point_spread = 2
        density=.5  # this is the spacing (used in linspace, np defaults to 50)
        dot_radius = 0.12  # (manim defaults to 0.08)
        vertical_space_ratio = 8
        scale_factor = 0.35

        grid = NumberPlane()
        self.add(grid)
        self.remove(grid)

        # create our 5x5 grid of dots ("perfect overlap")
        grid_coords = []
        for i in np.arange(-1, 1+density, density):
            for j in np.arange(-1, 1+density, density):
                grid_coords.append(np.array([float(i), float(j), float(0)], dtype=float))

        grid_dots_red = [Dot(i, radius=dot_radius).set_fill(MY_RED, opacity=1) for i in grid_coords]
        grid_dots_red_group = VGroup(*grid_dots_red)
        grid_dots_blue = [Dot(i, radius=dot_radius).set_fill(MY_BLUE, opacity=1) for i in grid_coords]
        grid_dots_blue_group = VGroup(*grid_dots_blue)

        self.add(grid_dots_blue_group, grid_dots_red_group)
        self.play(
            FadeIn(grid_dots_red_group),
            FadeIn(grid_dots_blue_group)
            )
        self.wait()

        # add jitter to the grid
        grid_jittered_coords_red = []
        for i in np.arange(-1, 1+density, density):
            for j in np.arange(-1, 1+density, density):
                x_jitter = point_spread*(np.random.rand(1)-.5)[0]
                y_jitter = point_spread*(np.random.rand(1)-.5)[0]
                grid_jittered_coords_red.append(np.array([i+x_jitter, j+y_jitter, 0]))

        grid_jittered_coords_blue = []
        for i in np.arange(-1, 1+density, density):
            for j in np.arange(-1, 1+density, density):
                x_jitter = 2*(np.random.rand(1)-.5)[0]
                y_jitter = 2*(np.random.rand(1)-.5)[0]
                grid_jittered_coords_blue.append(np.array([i+x_jitter, j+y_jitter, 0]))

        transition_anim = []
        for i, dot in enumerate(grid_dots_blue):
            transition_anim.append(dot.animate.move_to(grid_jittered_coords_blue[i]))

        for i, dot in enumerate(grid_dots_red):
            transition_anim.append(dot.animate.move_to(grid_jittered_coords_red[i]))

        self.play(
            AnimationGroup(*transition_anim)
            )
        self.wait()
        
        scale_down_anim_red = [dot.animate.scale(scale_factor) for dot in grid_dots_red]
        scale_down_anim_blue = [dot.animate.scale(scale_factor) for dot in grid_dots_blue]
        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.set(width=4),
            AnimationGroup(*scale_down_anim_blue),
            AnimationGroup(*scale_down_anim_red)
            )
        self.wait()


         # Find a red dot that is close to (0, 0, 0)
        distance_vectors = np.abs(np.repeat(np.array([[0, 0, 0]]), len(grid_jittered_coords_red), axis=0) - np.stack(grid_jittered_coords_red))
        distances = np.sqrt(distance_vectors[:, 0]**2 + distance_vectors[:, 1]**2 )
        closest_red_dot_idx = np.argmin(distances)
        red_dot_coord = grid_jittered_coords_red[closest_red_dot_idx]

        # Find blue dots that close to the selected red dot (N=5?)
        distance_vectors = np.abs(np.repeat(np.expand_dims(grid_jittered_coords_red[closest_red_dot_idx], 0), len(grid_jittered_coords_blue), axis=0) - np.stack(grid_jittered_coords_blue))
        distances = np.sqrt(distance_vectors[:, 0]**2 + distance_vectors[:, 1]**2 )
        blue_dots_by_distance = np.argsort(distances)
        blue_dot_coords = [grid_jittered_coords_blue[i] for i in blue_dots_by_distance]

        # draw arrows from red dots to blue dots
        # arrows = [Arrow(start=red_dot_coord, end=dest_coord, max_stroke_width_to_length_ratio=1) for dest_coord in blue_dot_coords[:5]]
        arrows = [Line(start=red_dot_coord, end=dest_coord, stroke_width=1, buff=.037) for dest_coord in blue_dot_coords[:5]]

        self.play(
            *[GrowFromPoint(arrow, red_dot_coord) for arrow in arrows]
            )
        self.wait()

        # do this for all the red dots
        arrows_all = []
        arrows_all_anim = []
        for curr_red_dot_coord in grid_jittered_coords_red:
            # Find blue dots that close to the selected red dot (N=5?)
            if np.allclose(curr_red_dot_coord, red_dot_coord):
                # skip the original above!
                continue
            distance_vectors = np.abs(np.repeat(np.expand_dims(curr_red_dot_coord, 0), len(grid_jittered_coords_blue), axis=0) - np.stack(grid_jittered_coords_blue))
            distances = np.sqrt(distance_vectors[:, 0]**2 + distance_vectors[:, 1]**2 )
            blue_dots_by_distance = np.argsort(distances)
            blue_dot_coords = [grid_jittered_coords_blue[i] for i in blue_dots_by_distance]

            # draw arrows from red dots to blue dots, use grey so we aren't overwhelming the viewer
            curr_arrows = [Line(start=curr_red_dot_coord, end=dest_coord, stroke_color=GREY, stroke_width=1, buff=.033) for dest_coord in blue_dot_coords[:5]]
            arrows_all.extend(curr_arrows)
            arrows_all_anim.extend([GrowFromPoint(arrow, curr_red_dot_coord) for arrow in curr_arrows])
        
        self.play(
            AnimationGroup(*arrows_all_anim)
            )
        self.wait(2) # time for arrows to stay alive

        # "Interpolate using pointnet" -- give a blue dot where the red dots are
        # explain how pointnet uses local features to interpolate this point

        interpolated_dots = [Dot(i, radius=dot_radius*scale_factor).set_fill(MY_BLUE, opacity=1) for i in grid_jittered_coords_red]
        interpolated_dots_group = VGroup(*interpolated_dots)

        self.play(
            FadeOut(grid_dots_red_group)
            )
        self.wait()

        self.play(
            FadeIn(interpolated_dots_group)
            )

        # highlight our new interpolated dots
        self.play(
            interpolated_dots_group.animate.set_fill(MY_YELLOW, opacity=1)
            )

        self.play(
            interpolated_dots_group.animate.set_fill(MY_BLUE, opacity=1)
            )
        self.wait()

        # return to normal
        scale_up_anim_red = [dot.animate.scale(1/scale_factor) for dot in grid_dots_red]
        scale_up_anim_interpolated = [dot.animate.scale(1/scale_factor) for dot in interpolated_dots]
        self.play(
            *[FadeOut(arrow) for arrow in arrows],
            *[FadeOut(arrow) for arrow in arrows_all],
            FadeIn(grid_dots_red_group),
            FadeOut(grid_dots_blue_group),
            Restore(self.camera.frame),
            AnimationGroup(*scale_up_anim_interpolated),
            AnimationGroup(*scale_up_anim_red)
            )
        self.wait()

        # mirror previous scene - convert to vertical line, show eqns, etc
        self.play(
            interpolated_dots_group.animate.shift(np.array([-3.5, 0, 0])),
            grid_dots_red_group.animate.shift(np.array([3.5, 0, 0]))
            )
        self.wait()

        self.play(
            interpolated_dots_group.animate.shift(np.array([3.5, 0, 0])),
            grid_dots_red_group.animate.shift(np.array([-3.5, 0, 0]))
            )
        self.wait()

        # convert to vertical lines
        num_dots = len(interpolated_dots)
        print("Num dots: ", num_dots)

        dots_vertical_coords = []
        for i in np.linspace(-num_dots/vertical_space_ratio, num_dots/vertical_space_ratio, num_dots):
            dots_vertical_coords.append([0, i, 0])

        interpolated_dots_vertical = [Dot(i, radius=dot_radius/2).set_fill(MY_BLUE, opacity=1) for i in dots_vertical_coords]
        red_dots_vertical = [Dot(i, radius=dot_radius/2).set_fill(MY_RED, opacity=1) for i in dots_vertical_coords]
        interpolated_dots_vertical_group = VGroup(*interpolated_dots_vertical)
        red_dots_vertical_group = VGroup(*red_dots_vertical)

        go_vertical_anim = []
        for i in range(num_dots):
            go_vertical_anim.append(ReplacementTransform(interpolated_dots[i], interpolated_dots_vertical[i])),
            go_vertical_anim.append(ReplacementTransform(grid_dots_red[i], red_dots_vertical[i]))

        self.play(
            AnimationGroup(*go_vertical_anim))

        # shift the dot sets
        self.play(
            interpolated_dots_vertical_group.animate.shift(np.array([-3.0, 0, 0])),
            red_dots_vertical_group.animate.shift(np.array([3.0, 0, 0]))
            )
        self.wait()

        vertical_expression_tex = MathTex(r'I_x = f(H_x)')
        vertical_expression_tex_2 = MathTex(r'H_x = g(I_x), g = f^{-1}')

        # write out our equation for kicks
        self.play(Write(vertical_expression_tex))
        self.wait()

        self.play(vertical_expression_tex.animate.shift(np.array([0, 1, 0])))
        self.wait()
        self.play(Write(vertical_expression_tex_2))
        self.wait()

        # harmonize dots
        self.play(interpolated_dots_vertical_group.animate.set_fill(MY_RED, opacity=1))
        self.wait()

        # remove tex
        self.play(
            vertical_expression_tex.animate.shift(np.array([0, 10, 0])),
            vertical_expression_tex_2.animate.shift(np.array([0, 10, 0])))
        self.wait()
        self.remove(vertical_expression_tex)
        self.remove(vertical_expression_tex_2)

        self.play(
            interpolated_dots_vertical_group.animate.shift(np.array([3.0, 0, 0])),
            red_dots_vertical_group.animate.shift(np.array([-3.0, 0, 0]))
            )
        self.wait()

        # put them back in the jittered grid
        # need to reset after a ReplacementTransform
        grid_jit_dot_red_h = [Dot(i, radius=dot_radius).set_fill(MY_RED, opacity=1) for i in grid_jittered_coords_red]
        grid_jit_dot_blue_h = [Dot(i, radius=dot_radius).set_fill(MY_RED, opacity=1) for i in grid_jittered_coords_red]

        go_back_anim = []
        for i in range(num_dots):
            go_back_anim.append(ReplacementTransform(interpolated_dots_vertical[i], grid_jit_dot_blue_h[i]))
            go_back_anim.append(ReplacementTransform(red_dots_vertical[i], grid_jit_dot_red_h[i]))

        self.play(
            AnimationGroup(*go_back_anim))
        self.wait()

        # fade out the dots show the overlapping unharmonized scans
        rect1_red = Rectangle(color=MY_RED, height=2, width=16).set_fill(MY_RED, opacity=0.5)
        rect1_red2 = Rectangle(color=MY_RED, height=2, width=16).set_fill(MY_RED, opacity=1)
        rect2_blue = Rectangle(color=MY_BLUE, height=10, width=2).set_fill(MY_BLUE, opacity=0.5)
        rect2_red = Rectangle(color=MY_RED, height=10, width=2).set_fill(MY_RED, opacity=1)

        self.play(
            VGroup(*grid_jit_dot_blue_h).animate.scale(.5),
            VGroup(*grid_jit_dot_red_h).animate.scale(.5))

        self.play(
            FadeOut(VGroup(*grid_jit_dot_blue_h)),
            FadeOut(VGroup(*grid_jit_dot_red_h)),
            FadeIn(rect2_blue),
            FadeIn(rect1_red)
            )
        self.wait()

        # harmonize the blue scan
        self.play(
            ReplacementTransform(rect1_red, rect1_red2),
            ReplacementTransform(rect2_blue, rect2_red)
            )
        self.wait()

        # fade out
        self.play(
            FadeOut(rect2_red),
            FadeOut(rect1_red2))
        self.wait()

        # FIN

class E_Results(Scene):
    def construct(self):
        title_screen = ImageMobject("results.png")
        self.add(title_screen)


        self.play(
            FadeIn(title_screen))
        self.wait()
        self.wait(4)
        self.play(
            FadeOut(title_screen))



class F_ResultsShift(Scene):
    def construct(self):
        title_screen = ImageMobject("results_shift.png")
        self.add(title_screen)


        self.play(
            FadeIn(title_screen))
        self.wait()
        self.wait(4)
        self.play(
            FadeOut(title_screen))
