
from loguru import logger
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor, AutoProcessor

class HFAutoClassIterator():

    def __init__(
        self,
        hf_repo_name,
        auto_classes = [AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor, AutoProcessor],
        cache_dir = None,
        verbose = True
    ):
        self.hf_repo_name = hf_repo_name
        self.auto_classes = auto_classes
        self.cache_dir = cache_dir
        self.verbose = verbose

    # return the successful autoclass objects
    def get_valid_auto_class_objects(self):
        auto_class_object_list = []
        for auto_class in self.auto_classes:
            try:
                auto_class_object_list.append(auto_class.from_pretrained(self.hf_repo_name, cache_dir=self.cache_dir))
            except:
                pass
        
        # remove repetitive objects
        auto_class_type_set = set()

        for autoclass in auto_class_object_list:
            if autoclass.__class__.__name__ in auto_class_type_set:
                auto_class_object_list.remove(autoclass)
                continue
            auto_class_type_set.add(autoclass.__class__.__name__)
        
        logger.info(
            f"Found autoclass objects: {[autoclass.__class__.__name__ for autoclass in auto_class_object_list]}"
        )
        return auto_class_object_list
