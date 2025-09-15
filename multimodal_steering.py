from steer.vector_generators.vector_generators import BaseVectorGenerator
from steer.vector_appliers.vector_applier import BaseVectorApplier
from steer.datasets import prepare_train_dataset, prepare_generation_datasets
from steer.utils.multimodal_utils import process_multimodal_item, validate_multimodal_dataset, get_multimodal_input_type
import hydra
from omegaconf import DictConfig

@hydra.main(version_base='1.2',config_path='./hparams/Steer', config_name='config.yaml')
def main(top_cfg: DictConfig):

    print("Global Config:", top_cfg, "\n")
    # Prepare datasets
    # You can use the datasets defined in config.yaml or define your own here
    # train_datasets = {
    #     'reasoning':[
    #         {'question': '1 + 1 = ', 
    #         'matching':'</think>\n\n1 + 1 equals 2. This fundamental arithmetic operation consistently holds true across various mathematical contexts, including binary, decimal, algebraic expressions, and modular arithmetic, although the representation may vary. In standard arithmetic, the sum of two ones is always two.<｜end▁of▁sentence｜>', 
    #         'not_matching': "Alright, so I'm trying to figure out what 1 + 1 equals. Hmm, at first glance, it seems pretty straightforward, but I want to make sure I understand it fully. Let me think about how addition works. When you add two numbers, you're combining their quantities. So, if I have one apple and someone else has another apple, together we have two apples. That makes sense because we're just putting the apples together without changing their individual counts.\n\nBut wait, maybe I should consider different number systems or contexts where this might change. For example, in binary, which is the base-2 system, 1 + 1 equals 10. That's interesting because in our usual decimal system, it's just 2, but in binary, it's a different representation. So, the way we add numbers can vary depending on the base we're using.\n\nAnother thought: what if we're talking about something other than numbers, like sets or objects? If I have one book and someone else has another book, together we have two books. It's the same concept, just adding the quantities. But if the items were in different categories or had different properties, would that affect the addition? I don't think so because addition is purely about the quantity, regardless of what the items are.\n\nI also wonder about the history of addition. How did humans figure out that combining two quantities gives a sum? It must have been through counting and recognizing patterns. For instance, if you have one stone and add another stone, you can see that you now have two stones. This simple concept likely formed the basis of mathematical addition.\n\nWhat about in mathematics, specifically in algebra? If I have variables, say x + x, that simplifies to 2x. So, in that case, 1 + 1 would be 2. It's consistent with the basic arithmetic we learned earlier. But what if it's more complex, like adding fractions or decimals? For example, 1/2 + 1/2 equals 1, and 0.5 + 0.5 also equals 1. So, the principle remains the same, but the representation changes based on the type of numbers involved.\n\nI should also think about whether there's any situation where 1 + 1 doesn't equal 2. In standard mathematics, across all number systems, 1 + 1 equals 2. Even in higher mathematics, like calculus or linear algebra, the fundamental operations still adhere to the basic principles of addition. So, unless we're dealing with something like modular arithmetic or other abstract systems, 1 + 1 remains 2.\n\nWait, in modular arithmetic, 1 + 1 modulo 2 would be 0. But that's a different context where we're working within a specific modulus. So, it's still 2 in the usual sense, but modulo 2, it's 0. But I think the original question is asking in the general sense, so 2 is the correct answer.\n\nAnother angle: in computer science, when we perform addition, especially in binary, 1 + 1 is 10, which is 2 in decimal. So, the result is the same, just represented differently. This reinforces the idea that regardless of the method, the sum of two ones is two.\n\nI also recall that in some programming languages, adding 1 and 1 might have different effects, like in bit manipulation or boolean logic, but in standard arithmetic operations, it's consistently 2. So, unless specified otherwise, 1 + 1 equals 2.\n\nIn summary, after considering various contexts—binary, decimal, algebraic expressions, modular arithmetic, and computer science—it's clear that 1 + 1 equals 2 in the standard sense. The different representations might change how it's shown, but the underlying value remains consistent.\n</think>\n\n1 + 1 equals 2. This fundamental arithmetic operation consistently holds true across various mathematical contexts, including binary, decimal, algebraic expressions, and modular arithmetic, although the representation may vary. In standard arithmetic, the sum of two ones is always two.<｜end▁of▁sentence｜>"
    #         }
    #     ]
    # }
    train_datasets = {
        'reasoning':[
            {'question': '1 + 1 = ', 
             'image': 'path/to/your/image.jpg',
             'matching':'</think>\n\n1 + 1 equals 2. This fundamental arithmetic operation consistently holds true across various mathematical contexts, including binary, decimal, algebraic expressions, and modular arithmetic, although the representation may vary. In standard arithmetic, the sum of two ones is always two.<｜end▁of▁sentence｜>', 
             'not_matching': "Alright, so I'm trying to figure out what 1 + 1 equals. Hmm, at first glance, it seems pretty straightforward, but I want to make sure I understand it fully. Let me think about how addition works. When you add two numbers, you're combining their quantities. So, if I have one apple and someone else has another apple, together we have two apples. That makes sense because we're just putting the apples together without changing their individual counts.\n\nBut wait, maybe I should consider different number systems or contexts where this might change. For example, in binary, which is the base-2 system, 1 + 1 equals 10. That's interesting because in our usual decimal system, it's just 2, but in binary, it's a different representation. So, the way we add numbers can vary depending on the base we're using.\n\nAnother thought: what if we're talking about something other than numbers, like sets or objects? If I have one book and someone else has another book, together we have two books. It's the same concept, just adding the quantities. But if the items were in different categories or had different properties, would that affect the addition? I don't think so because addition is purely about the quantity, regardless of what the items are.\n\nI also wonder about the history of addition. How did humans figure out that combining two quantities gives a sum? It must have been through counting and recognizing patterns. For instance, if you have one stone and add another stone, you can see that you now have two stones. This simple concept likely formed the basis of mathematical addition.\n\nWhat about in mathematics, specifically in algebra? If I have variables, say x + x, that simplifies to 2x. So, in that case, 1 + 1 would be 2. It's consistent with the basic arithmetic we learned earlier. But what if it's more complex, like adding fractions or decimals? For example, 1/2 + 1/2 equals 1, and 0.5 + 0.5 also equals 1. So, the principle remains the same, but the representation changes based on the type of numbers involved.\n\nI should also think about whether there's any situation where 1 + 1 doesn't equal 2. In standard mathematics, across all number systems, 1 + 1 equals 2. Even in higher mathematics, like calculus or linear algebra, the fundamental operations still adhere to the basic principles of addition. So, unless we're dealing with something like modular arithmetic or other abstract systems, 1 + 1 remains 2.\n\nWait, in modular arithmetic, 1 + 1 modulo 2 would be 0. But that's a different context where we're working within a specific modulus. So, it's still 2 in the usual sense, but modulo 2, it's 0. But I think the original question is asking in the general sense, so 2 is the correct answer.\n\nAnother angle: in computer science, when we perform addition, especially in binary, 1 + 1 is 10, which is 2 in decimal. So, the result is the same, just represented differently. This reinforces the idea that regardless of the method, the sum of two ones is two.\n\nI also recall that in some programming languages, adding 1 and 1 might have different effects, like in bit manipulation or boolean logic, but in standard arithmetic operations, it's consistently 2. So, unless specified otherwise, 1 + 1 equals 2.\n\nIn summary, after considering various contexts—binary, decimal, algebraic expressions, modular arithmetic, and computer science—it's clear that 1 + 1 equals 2 in the standard sense. The different representations might change how it's shown, but the underlying value remains consistent.\n</think>\n\n1 + 1 equals 2. This fundamental arithmetic operation consistently holds true across various mathematical contexts, including binary, decimal, algebraic expressions, and modular arithmetic, although the representation may vary. In standard arithmetic, the sum of two ones is always two.<｜end▁of▁sentence｜>"
               }
        ]
    }
    
    # 验证多模态数据集格式
    for dataset_name, dataset in train_datasets.items():
        if not validate_multimodal_dataset(dataset):
            print(f"Warning: Dataset {dataset_name} format validation failed!")
        else:
            print(f"Dataset {dataset_name} format validation passed!")
            # 显示数据集类型统计
            input_types = [get_multimodal_input_type(item) for item in dataset]
            type_counts = {}
            for input_type in input_types:
                type_counts[input_type] = type_counts.get(input_type, 0) + 1
            print(f"Input type distribution: {type_counts}")
    
    # train_datasets = prepare_train_dataset(top_cfg)
    # generation_datasets={
    #     'reasoning':[
    #         {'input': "9.8 or 9.11, which is bigger?"}
    #     ]
    # }
    # generation_datasets ={
    #     'reasoning':[
    #         [{"type": "text", "content": "Describe the image in detail."},
    #         {"type": "image", "content": "path/to/your/image.jpg"}]
                    
    #     ]
    # }
    generation_datasets ={
        'reasoning':[
            {"text": "Describe the image in detail.",
            "image": "path/to/your/image.jpg"}
                    
        ]
    }
    # generation_datasets = prepare_generation_datasets(top_cfg)

    # Generate Steering Vectors
    vector_generator = BaseVectorGenerator(top_cfg)
    vectors = vector_generator.multimodal_generate_vectors(train_datasets)

    # Apply Vectors to Model 
    vector_applier = BaseVectorApplier(top_cfg)
    print(vectors)
    for dataset in vectors.keys():
        print(f"Applying  {dataset} vectors to model ...")
        vector_applier.multimodal_apply_vectors(vectors[dataset])

    # vector_applier.apply_vectors()

    # Result Generation
    vector_applier.multimodal_generate(generation_datasets)

    # Resets the model to its initial state, clearing any modifications.
    vector_applier.model.reset_all()

if __name__=='__main__':
    main()
