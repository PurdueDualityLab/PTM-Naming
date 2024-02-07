import re
import json
import random

from loguru import logger


class Cleaner:
    def __init__(self, data: dict):
        self.data = data
        self.model_names = data.keys()

        self.uninformative_names = []
        self.cls_data = {}

    def clean(self) -> dict:
        self.uninformative_names = self.get_uninformative()
        self.implementation_names = self.get_implementation_names()
        # self.application_names = self.get_application_names()
        self.combined = self.get_combined_names()

        logger.info(f"Number of uninformative names: {len(self.uninformative_names)}")
        logger.info(f"Number of implementation names: {len(self.implementation_names)}")
        # logger.debug(f"implementation names: {self.implementation_names[:10]}")
        # logger.info(f"Number of application names: {len(self.application_names)}")
        # logger.debug(f"Application names: {self.application_names[:10]}")
        logger.info(f"Number of combined names: {len(self.combined)}")

        return self.cls_data   
    

    def get_uninformative(self) -> list:
        uninformative = []
        # Collect keys to be removed
        for key in self.data.keys():
            if self._is_uninformative(key):
                uninformative.append(key)
        
        # Remove the keys after collecting them
        for key in uninformative:
            del self.data[key]
            self.cls_data[key] = 'uninformative'
        # Assuming logger is defined and available in your context
        # logger.success(f'Removed number of uninformative keys: {len(uninformative)}')
        # logger.debug(f'Uninformative keys: {uninformative}')
        
        return uninformative


    def _is_uninformative(self, name: str) -> bool:
        '''
        Returns True if the value is uninformative
        '''
        import re
        uninform_pattern = r'^(autotrain|autonlp|modeleval|test|demo|my|dummy).*'

        if re.search(uninform_pattern, name.split('/')[-1], re.IGNORECASE):
            return True
        return False


    def get_implementation_names(self) -> list:
        '''
        Returns a list of implementation model names
        '''
        implementation_names = []
        for name in self.data.keys():
            if self._is_implementation(name):
                implementation_names.append(name)
        
        for key in implementation_names:
            self.cls_data[key] = 'implementation unit'
            del self.data[key]
        return implementation_names



    def _is_implementation(self, name: str) -> bool:
        '''
        Returns True if the model name is considered high quality based on its architecture.
        Architecture should be explicitly present in the name, taking into account normalization to
        handle hyphens and underscores in both the architecture and model names.
        '''
        model_name_part = name.split('/')[-1].lower()  # Work with the part after '/'

        architecture = self.data.get(name, "")
        architecture_name = self._extract_architecture_name(architecture)

        if architecture_name:
            # Generate a pattern that matches the architecture name with any non-alphanumeric characters removed
            # This allows "GPT-2" to match "gpt2" by ignoring hyphens in the comparison
            normalized_architecture_name = re.sub(r'[\W_]+', '', architecture_name).lower()
            pattern = re.compile(re.escape(normalized_architecture_name), re.IGNORECASE)

            # Check if the architecture name is present in the model name after normalization
            if pattern.search(re.sub(r'[\W_]+', '', model_name_part)):
                return True

        return False


    def _extract_architecture_name(self, architecture: str) -> str:
        '''
        Extracts the architecture name from a given architecture string.
        '''
        match = re.match(r'([A-Za-z0-9_]+)(?:[vV][1-9])?(?:For|Model|LMHead)', architecture)
        if match:
            return match.group(1).lower()  # Convert to lower case for case-insensitive comparison
        return ""


    def get_application_names(self) -> list:
        '''
        The rest of data are all application names
        '''
        application_names = []
        for key in self.data.keys():
            if self._is_application(key):
                application_names.append(key)

        for key in application_names:
            self.cls_data[key] = 'application goal'
            del self.data[key]
        return application_names
    

    def _is_application(self, name: str) -> bool:
        '''
        Returns True if the model name does not include the architecture in any part of its name.
        '''
        # Extract the part after '/' if present, as the model name part
        model_name_part = name.split('/')[-1] if '/' in name else name

        # Loop through all possible architectures to check presence in the model name
        for architecture in self.data.values():
            architecture_name = self._extract_architecture_name(architecture)
            # If the architecture name is found anywhere in the model name, return False
            if architecture_name and architecture_name.lower() in model_name_part.lower():
                return False
        # If no architecture is found in the model name, it's considered an application name
        return True

    
    def get_combined_names(self) -> list:
        '''
        Returns the rest of the model names
        '''
        combined_names = []
        for key in self.data.keys():
            combined_names.append(key)

        for key in list(self.data.keys()):
            self.cls_data[key] = 'combined pattern'
        return combined_names
    


    def get_model_names(self) -> list:
        '''
        Returns a list of model names
        '''
        return self.model_names
    

    def get_model_examples(self, num_of_examples) -> list:
        '''
        Returns a list of examples for the model
        '''
        rand_model_names = random.sample(list(self.model_names), num_of_examples)
        return rand_model_names
    
    
    def get_cls_data(self) -> dict:
        '''
        Returns the classification data
        '''
        return self.cls_data
    


def dataloader(file_path: str) -> dict:
    with open(file_path) as f:
        data = json.load(f)
    return data


def main():
    
    # Loading the filtered data (>=10 downloads, >1 models per architecture)
    data = dataloader('../model_collection/filtered_name_to_architecture.json')
    logger.info(f"Number of models: {len(data)}")
    cleaner = Cleaner(data)
    ############################
    # samples = cleaner.get_model_examples(5)
    # for sample in samples:
    #     logger.debug(f"Sample: {sample}")
    #     logger.debug(cleaner.data[sample])
    # # logger.debug(len(cleaner.get_model_names()))
    ############################
    cls_data = cleaner.clean()
    # logger.debug(cls_data)

    with open('cls_data.json', 'w') as f:
        json.dump(cls_data, f, indent=4)

    
    


if __name__ == '__main__':
    main()