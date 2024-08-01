from fastbook import *
from fastai.vision.widgets import *

class ImageClassifier:
    def __init__(self, path):
        self.path = Path(path)

    
    
class ImageDownloader(ImageClassifier) :

    def __init__(self, search_terms, path):
        super().__init__(path)
        self.search_terms = search_terms

    def download_images(self):
        for term in self.search_terms:
            dest = self.path / term
            dest.mkdir(parents=True, exist_ok=True)
            results = search_images_ddg(f'{term} images', max_images=10)
            for i, url in enumerate(results):
                try:
                    download_url(url, dest/f'{term}{i}.jpg')
                except Exception as e:
                    print(f"Error downloading {url}: {e}")

    def items_per_label(self):
        dict =  {}
        for l in self.search_terms:
            dict[f'{l}'] = len(get_image_files(self.base_path/l))
        print(dict)
        return dict
    
    def get_images(self):
        return get_image_files(self.path)
    
    
class DataProcessor(ImageClassifier):

    def __init__(self, path):
        super().__init__(path)

    def verify_images(self, images):
        return verify_images(images)
    
    def remove_failed_images(self, failed_images):
        failed_images.map(Path.unlink)

    def create_dataloaders(self, item_tfms, batch_tfms=None):
        datablock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=parent_label,
            item_tfms=item_tfms,
            batch_tfms=batch_tfms
        )
        return datablock.dataloaders(self.path)

    
    

    

if __name__ == "__main__":
    
    downloader = ImageDownloader(['red', 'green', 'logo'], 'apples')
    downloader.download_images()
    images = downloader.get_images()

    processor = DataProcessor('images')
    failed_images = processor.verify_images(images)
    processor.remove_failed_images(failed_images)
    dls = processor.create_dataloaders(item_tfms=Resize(128))

    for batch in dls.valid:
        print(batch)
    