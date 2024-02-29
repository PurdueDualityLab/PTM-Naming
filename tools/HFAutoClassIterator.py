
from loguru import logger
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor, AutoProcessor

class HFAutoClassIterator():

    def __init__(
        self,
        hf_repo_name,
        auto_classes = [AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor, AutoProcessor],
        cache_dir = None,
        verbose = True,
        trust_remote_code = False
    ):
        self.hf_repo_name = hf_repo_name
        self.auto_classes = auto_classes
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.trust_remote_code = trust_remote_code

    # return the successful autoclass objects
    def get_valid_auto_class_objects(self):
        auto_class_object_list = []
        err_report = {}
        for auto_class in self.auto_classes:
            try:
                auto_class_object_list.append(auto_class.from_pretrained(self.hf_repo_name, cache_dir=self.cache_dir, trust_remote_code=self.trust_remote_code))
            except Exception as emsg:
                err_report[auto_class.__name__] = emsg
                continue
        if auto_class_object_list == []:
            return err_report
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