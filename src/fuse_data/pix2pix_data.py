from typing import Tuple

from datasets import load_dataset


class Pix2Pix:
    def __init__(self, txt_tokenizer, image_processor):
        self.data = load_dataset("timbrooks/instructpix2pix-clip-filtered")["train"]

        self.txt_tokenizer = txt_tokenizer
        self.image_processor = image_processor

    def grouped_by_edit(self):
        # create dict {edit0: [idx0, idx0, ...], edit1: ...}
        all_edits = self.data["edit_prompt"]
        edit_to_idx = {}
        for idx, edit in enumerate(all_edits):
            edit_to_idx.setdefault(edit, []).append(idx)
        return edit_to_idx

    def grouped_by_prompt(self):
        # create dict {edit0: [idx0, idx0, ...], edit1: ...}
        all_edits = self.data["original_prompt"]
        prompt_to_idx = {}
        for idx, edit in enumerate(all_edits):
            prompt_to_idx.setdefault(edit, []).append(idx)
        return prompt_to_idx


    def _process_image(self, image):
        if image is None:
            return None
        image = self.image_processor(image)
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item) -> Tuple:
        el = self.data[item]
        input_image = el["original_image"]
        edit = el["edit_prompt"]
        output_image = el["edited_image"]

        edit = self.txt_tokenizer(edit).squeeze()
        input_image = self._process_image(input_image)
        right_txt = self.txt_tokenizer("").squeeze()
        output_image = self._process_image(output_image)
        return edit, input_image, right_txt, output_image







if __name__ == '__main__':
    from open_clip import get_tokenizer
    from fuse_clip.fuse_clip_preprocess import get_fuse_clip_image_preprocess
    import matplotlib.pyplot as plt
    import textwrap

    def decode_tokens(tokenizer, tokens):
        return tokenizer.decode(tokens[tokens!=0][1:-1].squeeze().tolist())

    txt_tokenizer = get_tokenizer("fuse-clip-titok", context_length=180)
    image_processor = get_fuse_clip_image_preprocess(train=True)

    dataset = Pix2Pix(txt_tokenizer=txt_tokenizer, image_processor=image_processor)

    grouped_by_edit = dataset.grouped_by_edit()
    grouped_by_prompt = dataset.grouped_by_prompt()
    exit()

    # plot examples
    nrows = 10
    fig, axs = plt.subplots(nrows, 4, figsize=(10, nrows*3))
    for i in range(nrows*2):
        row = i // 2
        col = 0 if i % 2 == 0 else 2
        edit, input_image, right_txt, output_image = dataset[i]
        edit = decode_tokens(txt_tokenizer, edit)
        axs[row, col].imshow(input_image.permute(1, 2, 0))
        axs[row, col].set_title(textwrap.fill(edit, 40), fontsize=8)
        axs[row, col+1].imshow(output_image.permute(1, 2, 0))
    [ax.axis("off") for ax in axs.ravel()]
    plt.tight_layout()
    plt.show()

    print("done")