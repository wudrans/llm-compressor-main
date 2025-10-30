from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation
import os
import torch

# 早期的Qwen3版本
previous_Qwen3 = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]


input_message = '''总结文章大意：\n    有一个豆荚，里面有五粒豌豆。豆荚和豌豆都是绿的，豌豆就以为整个世界都是绿的。豆荚在生长，豌豆也在生长。豌豆按照它们在家庭里的地位，坐成一排。太阳在外边照着，把豆荚晒得暖洋洋的。这里既温暖，又舒适;白天明亮，夜间黑暗。豌豆坐在那儿越长越大，它们想，我们得做点儿事情啊。\n\n　　“难道我们永远就在这儿坐下去吗?”它们中的一个问，“老这样坐下去，我恐怕会变得僵硬起来。我似乎觉得外面发生了一些事情——我有这种预感!”\n\n　　许多天过去了。豆荚变黄了，这几粒豌豆也变黄了。“整个世界都变黄啦!”它们说。\n\n　　忽然，它们觉得豆荚震动了一下。豆荚被摘下，跟许多别的丰满的豆荚在一起，溜到一个口袋里去了。\n\n　　“我们不久就要被打开了!”豌豆们说。于是它们就等待这件事情的到来。\n\n　　“我倒想要知道，我们之中谁会走得最远!”最小的一粒豆说，“是的，事情马上就要揭晓了。”\n\n　　啪!豆荚裂开来了。那五粒豆子全都躺在一个孩子的手中。这个孩子紧紧地捏着它们，说可以当作玩具枪的子弹用。他马上把第一粒豆子装进去，把它射了出去。\n\n　　“现在我要飞向广大的世界里去了!如果你能捉住我，就请你来吧!”第一粒豌豆说完就飞走了。\n\n　　“我，”第二粒说，“我将直接飞进太阳里去。这才像一粒豌豆呢，而且与我的身份非常相称!”于是，它也飞走了。\n\n　　“我们到了哪儿，就在哪儿住下来，”其余的两粒说，“不过，我们还得向前滚。”在没有被装进玩具枪之前，它们从小孩的手中滑落到地上，打起滚来。但这两粒豆最终还是被装进玩具枪里去了。它们说：“我们才会射得最远呢!”\n\n　　“该怎么样就怎么样吧!”最后的那一粒说。它被射到空中，落到顶楼窗子下面的一块旧板子上，正好钻进一个长满了青苔的裂缝里。青苔把它裹起来，它躺在那儿真可以说成了一个囚犯。\n\n　　“该怎么样就怎么样!”这粒豆说。\n\n　　在这个小小的顶楼里住着一个穷苦的女人。她有一个独生女儿，身体非常虚弱，躺在床上一整年了。小女孩安静地、耐心地整天在家里躺着，而她的母亲每天到外面去挣点儿生活费。\n\n　　春天的一个早晨，当母亲准备出去工作的时候，太阳温和地从那个小窗子射进来，一直射到地上。\n\n　　小女孩望着最低的那块窗玻璃说：“有个绿东西从窗玻璃旁边探出头来，它是什么呢?”\n\n　　母亲向窗边走去，把窗户打开一半。“啊!”她说，“我的天，原来是一粒小豌豆在这里生了根，还长出小叶子来了。它怎么钻进这个隙缝里去的?你现在有一个小花园了!”\n\n　　母亲把小女孩的床搬得更靠近窗子，好让她看到这粒正在生长着的豌豆。\n\n　　“妈妈，我觉得我好了一些!”晚上，这个小女孩说，“太阳今天在我身上照得怪暖和的。这粒豆子长得好极了，我也会好起来的;我能爬起来，走到温暖的太阳光中去。”\n\n　　虽然母亲不相信，但她还是仔细地用一根小棍子把这植物支起来，好使它不至于被风吹断，因为它使女儿对生命产生了愉快的想象。她从窗台上牵了一根绳子到窗框的上端去，使这棵豌豆苗可以盘绕着它向上生长。\n\n　　它的确在向上长——人们每天都可以看到它在生长。\n\n　　“真的，它现在要开花了!”这个母亲慢慢开始相信，她的孩子会好起来。她记起最近这孩子讲话时，要比以前愉快得多，而且最近几天她也能自己爬起来，直直地坐在床上，用兴奋的眼光望着这一粒豌豆所形成的小花园。一星期以后，小女孩第一次能够坐一整个钟头。她快乐地坐在温暖的太阳光里。窗子打开了，她面前是一朵盛开的、粉红色的豌豆花。小姑娘低下头来，轻轻地吻了一下它柔嫩的叶子。这一天简直像一个节日。\n\n　　其余的几粒豌豆呢?曾经想飞到广大世界里去的那一粒，它落到了屋顶的水笕里，被一只鸽子吃掉了。那两粒在地上打滚的豆子也没有走多远，也被鸽子吃掉了。它们还算有些实际的用途。那本来想飞进太阳里去的豌豆，却落到了水沟里，在脏水里躺了好几个星期，而且涨得大大的。\n\n　　“我胖得够美了!”这粒豌豆说，“我胖得要爆裂开来了。我想任何豌豆从来不曾、也永远不会达到这种地步的。我是五粒豌豆中最了不起的一粒。”\n\n　　此刻，顶楼窗子旁那个小女孩——她的脸上洋溢着健康的光彩，她的眼睛发着亮光——正注视着豌豆花，快乐地微笑着。心里充满了感激。"'''
# 预处理成标准格式，如<|im_start|>user... <|im_end|>
def preprocess1(example, tokenizer, model_path):
    basename = os.path.basename(model_path)
    if basename in previous_Qwen3:
        # Qwen3-1.7B模型使用enable_thinking, Default is True.
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False, # 是否要将文本转换为token ID序列
            add_generation_prompt=True, # 表示添加生成提示符,通常用于指示模型开始生成文本
            enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
        )
    else:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False, # 是否要将文本转换为token ID序列
            add_generation_prompt=True, # 表示添加生成提示符,通常用于指示模型开始生成文本
        )
    result = {"text": text}
    # print(result)
    return result
    '''
    text: 
        <|im_start|>user
        请写一篇文章：我的妈妈，不少于1000字
        <|im_end|>
        <|im_start|>assistant
        ......
        <|im_end|>
        <|im_start|>user
        ......
        <|im_end|>
        ....
    '''


def model_test(model, tokenizer, input_message, max_new_tokens):
    input_ids = tokenizer(input_message, return_tensors="pt").to(model.device)
    input_length = len(input_ids.input_ids[0])

    output = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    result = tokenizer.decode(output[0][input_length:])
    # print(result)
    return result


# Select model and load it.
MODEL_ID = "/data8T/models/LLM/Qwen/Qwen3-0.6B"
saved_root = "/data8T/models/LLM/Qwen_quantize-awq-sym"
saved_path = os.path.join(saved_root, os.path.basename(MODEL_ID))
os.makedirs(saved_path, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# input_message = "你好，我是"
max_new_tokens = 256
original_ouput = model_test(model, tokenizer, input_message, max_new_tokens)

# Select calibration dataset.
DATASET_PATH="/data8T/Text/HuggingFaceH4/ultrachat_200k/data"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 2
MAX_SEQUENCE_LENGTH = 512

# =============== Load dataset and preprocess.===============
# 第一种方法是直接下载数据集到缓存中，需要连接VPN才可以下载，
# ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
# **************************************************************
# 第二种方法先下载到缓存中，再从指定路径加载，两种方法得到的ds已验证相同
tmp_path = os.path.join(DATASET_PATH, f"{DATASET_SPLIT}-*.parquet")
# 使用 streaming 模式，不会一次性加载所有数据到内存
iterable_ds = load_dataset("parquet", 
                  data_files=tmp_path,streaming = True)
print(iterable_ds)
# 取前256个样本并转换为常规Dataset
data_list = []
for i, example in enumerate(iterable_ds['train']):
    if i >= NUM_CALIBRATION_SAMPLES:
        break
    data_list.append(example)
ds = Dataset.from_list(data_list) # 转换为常规Dataset

# 列出几个关键词包括：features: ['prompt', 'prompt_id', 'messages']
# prompt:提问词
# messages:是一个列表，通常包含多轮对话，每轮对话是一个字典
# [{'content':"....", 'role':'user'}, {'content':"....", 'role':'assistant'}, ....]
# 如何从ds从取样本
# ds[0]代表第0个样本，是一个dict,包含的key:prompt,prompt_id, messages
# print(ds[0])

ds = ds.shuffle(seed=42)

# 预处理成标准格式，如<|im_start|>user... <|im_end|>
def preprocess(example):
    text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False)
    result = {"text": text}
    # print(result)
    return result
    '''
    text: 
        <|im_start|>user
        请写一篇文章：我的妈妈，不少于1000字
        <|im_end|>
        <|im_start|>assistant
        ......
        <|im_end|>
        <|im_start|>user
        ......
        <|im_end|>
        ....
    '''
# ds = ds.map(preprocess) is same to 
ds = ds.map(lambda example: preprocess1(example, tokenizer, MODEL_ID))
# print(ds)
# print(ds[0])
# exit()
# Dataset({
#     features: ['prompt', 'prompt_id', 'messages', 'text'],
#     num_rows: 2
# })


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )
    '''
    return {'input_ids': tensor([[151644,   8948,    198,  37029, 104811, 102104, 151645,    198, 151644,
        872,    198,  14880,  61443, 116562,   5122,  97611, 101935, 151645,
        198, 151644,  77091,    198, 151667,    271, 151668,    271]],
    device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1]], device='cuda:0')}
    '''


# Configure the quantization algorithm to run.
# NOTE: vllm currently does not support asym MoE, using symmetric here
recipe = [
    AWQModifier(
        # ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
        ignore=[
            "lm_head", 
            "embed_tokens",  # 添加embedding层
            "re:.*norm.*",   # 忽略所有norm层
            "re:.*mlp.gate$", 
            "re:.*mlp.shared_expert_gate$",
            "re:.*attention.*output.*",  # 考虑忽略注意力输出层
        ],
        scheme="W4A16",
        targets=["Linear"],
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer(input_message, return_tensors="pt").to(model.device)
input_length = len(input_ids.input_ids[0])

output = model.generate(**input_ids, max_new_tokens=max_new_tokens)
print(tokenizer.decode(output[0][input_length:]))

print("=================quantize output=====================\n\n")
quantize_ouput = model_test(model, tokenizer, input_message, max_new_tokens)
print(quantize_ouput)

print("=================original output=====================\n\n")
print(original_ouput)





# Save to disk compressed.
model.save_pretrained(saved_path, save_compressed=True)
tokenizer.save_pretrained(saved_path)
print("sucessfuly saved to %s" % saved_path)
