import imp
from operator import truediv
import modules.scripts as scripts
import gradio as gr
import os

from modules import images
from modules.processing import process_images, Processed, StableDiffusionProcessingImg2Img
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):  

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "Force symmetry"

# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):
        return not is_img2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        h_symmetry = gr.Checkbox(False, label="Horizontal symmetry")
        v_symmetry = gr.Checkbox(False, label="Vertical symmetry")
        alt_symmetry_mode = gr.Checkbox(False, label="Alt. symmetry method (blending)")
        every_n_steps = gr.Number(
            1,
            label="Apply every n steps",
            precision=0,
            interactive=True
            )
        skip_last_n_steps = gr.Number(
            1,
            label="Skip last n steps",
            precision=0,
            interactive=True
            )
        return [h_symmetry, v_symmetry, every_n_steps, skip_last_n_steps, alt_symmetry_mode]

    def run(self, p, h_symmetry, v_symmetry, every_n_steps, skip_last_n_steps, alt_symmetry_mode):
        import numpy as np
        from PIL import Image, ImageOps

        def get_steps_schedule(total_steps, every_n_step, skip_last_n):
            if skip_last_n >= total_steps:
                skip_last_n = total_steps - 1
            schedule = []
            total_steps -= 1
            img2img_repeats = (total_steps - skip_last_n) // every_n_step
            for it in range(img2img_repeats):
                schedule.append(every_n_step)
            s = (total_steps - skip_last_n) % every_n_step
            if s:
                schedule.append(s)
            if skip_last_n:
                schedule.append(skip_last_n)
            return schedule

        def apply_symmetry_alt(img, flip_x, flip_y):
            if flip_x:
                img_fl_x = ImageOps.mirror(img)
                img = Image.blend(img, img_fl_x, 0.5)
            if flip_y:
                img_fl_y = ImageOps.flip(img)
                img = Image.blend(img, img_fl_y, 0.5)
            return img
        
        def apply_symmetry(img, flip_x, flip_y):
            if flip_x:
                box = (
                    0,
                    0,
                    img.width / 2,
                    img.height
                )
                img_fl_x = ImageOps.mirror(img.crop(box))
                img.paste(img_fl_x, (img.width // 2, 0))
            if flip_y:
                box = (
                    0,
                    0,
                    img.width,
                    img.height / 2
                )
                img_fl_y = ImageOps.flip(img.crop(box))
                img.paste(img_fl_y, (0, img.height // 2))
            return img

        assert every_n_steps >= 1
        assert skip_last_n_steps >= 0
        total_steps = p.steps
        state.sampling_steps = total_steps

        apply_symmetry_func = apply_symmetry_alt if alt_symmetry_mode else apply_symmetry

        steps_schedule = get_steps_schedule(
            total_steps,
            every_n_steps,
            skip_last_n_steps)
        print(steps_schedule)

        state.job_count = len(steps_schedule) + 1
        p.steps = 1
        proc = process_images(p)
        for i in range(len(proc.images)):
            proc.images[i] = apply_symmetry_func(proc.images[i], h_symmetry, v_symmetry)
            for iter_num, img2img_iter in enumerate(steps_schedule):
                is_last_iter = (iter_num == len(steps_schedule) - 1)
                img2img_processing = StableDiffusionProcessingImg2Img(
                    init_images=proc.images,
                    resize_mode=0,
                    denoising_strength=1,
                    mask=None,
                    mask_blur=4,
                    inpainting_fill=0,
                    inpaint_full_res=True,
                    inpaint_full_res_padding=0,
                    inpainting_mask_invert=0,
                    sd_model=p.sd_model,
                    outpath_samples=p.outpath_samples,
                    outpath_grids=p.outpath_grids,
                    prompt=p.prompt,
                    styles=p.styles,
                    seed=proc.seed,
                    subseed=proc.subseed,
                    subseed_strength=p.subseed_strength,
                    seed_resize_from_h=p.seed_resize_from_h,
                    seed_resize_from_w=p.seed_resize_from_w,
                    #seed_enable_extras=p.seed_enable_extras,
                    sampler_index=p.sampler_index,
                    batch_size=p.batch_size,
                    n_iter=p.n_iter,
                    steps=img2img_iter,
                    cfg_scale=p.cfg_scale,
                    width=p.width,
                    height=p.height,
                    restore_faces=p.restore_faces,
                    tiling=p.tiling,
                    do_not_save_samples=p.do_not_save_samples if is_last_iter else True,
                    do_not_save_grid=p.do_not_save_grid if is_last_iter else True,
                    extra_generation_params=p.extra_generation_params,
                    overlay_images=p.overlay_images,
                    negative_prompt=p.negative_prompt,
                    eta=p.eta
                    )
                
                proc = process_images(img2img_processing)
                if not is_last_iter:
                    for i in range(len(proc.images)):
                        proc.images[i] = apply_symmetry(proc.images[i], h_symmetry, v_symmetry)
        return proc