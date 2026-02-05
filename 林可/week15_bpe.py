from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# 初始化BPE分词器
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 定义训练器，设置合并规则和特殊符号
trainer = BpeTrainer(
    vocab_size=1000,
    min_frequency=1,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
)

# 训练语料（即你提供的原文）
train_corpus = [
    "编者按：今年冬季让叠穿打造你的完美身材，越穿越瘦是可能。哪怕是不了解流行，也要对时尚又显瘦的叠穿造型吸引，现在就开始行动吧！搭配Tips：亮红色的皮外套给人光彩夺目的感觉，内搭短版的黑色T恤，露出带有线条的腹部是关键，展现你健美的身材。 搭配Tips：简单款型的机车装也是百搭的单品，内搭一条长版的连衣裙打造瘦身的中性装扮。软硬结合的mix风同样备受关注。 搭配Tips：贴身的黑色装最能达到瘦身的效果，即时加上白色的长外套也不会发福。长款的靴子同样很好的修饰了你的小腿线条。 搭配Tips：高腰线的抹胸装很有拉长下身比例的效果，A字形的荷叶摆同时也能掩盖腰部的赘肉。外加一件短款的羽绒服，配上贴腿的仔裤，也很修长。"
]

# 训练BPE分词器
tokenizer.train_from_iterator(train_corpus, trainer=trainer)

# 处理分词输出格式（移除特殊符号，仅保留分词结果）
tokenizer.post_processor = TemplateProcessing(
    single="$A",
    special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
)

# 对原文进行分词
text = train_corpus[0]
output = tokenizer.encode(text)

# 输出分词结果（用▁分隔，符合BPE常见输出格式）
print("▁".join(output.tokens))
