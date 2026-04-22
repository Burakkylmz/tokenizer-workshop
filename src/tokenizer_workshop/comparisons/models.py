from __future__ import annotations

# dataclass:
# Veri taşıyan sınıfları daha kısa, temiz ve okunabilir yazmak için kullanılır.
# Normalde her sınıf için tek tek __init__, __repr__, __eq__ gibi metodlar yazmak gerekir.
# @dataclass bunu otomatik üretir ve özellikle "veri modeli" sınıflarında kodu sadeleştirir.
from dataclasses import dataclass, field


# ============================================================
# TokenizerMetrics
# ============================================================
# Bu sınıfın görevi:
# Tek bir tokenizer çalıştırıldıktan sonra elde edilen sayısal metrikleri düzenli bir yapıda tutmaktır.
#
# Önemli nokta:
# Burada token'ların kendisini değil, token'lar üzerinden hesaplanan "analiz bilgileri" saklanır.
#
# Örnek:
# tokens = ["Hello", "world!", "Tokenization", "is", "fun."]
#
# Bu token listesi üzerinden hesaplanabilecek metrikler:
# - toplam token sayısı
# - tekil token sayısı
# - ortalama token uzunluğu
# - minimum token uzunluğu
# - maksimum token uzunluğu
# - ortalama karakter sayısı / token
# - compression ratio
# - top 10 token
# - token length distribution
# - average chars per token
# - efficiency score
# - unknown token sayısı
# - latency
#
# Metrikleri ayrı bir sınıfta toplamak iyi bir tasarım tercihidir çünkü:
# 1) token listesi ile analiz bilgisi birbirinden ayrılmış olur
# 2) ileride yeni metrik eklemek kolaylaşır
# 3) CompareManager daha temiz kalır
# 4) test etmek kolaylaşır
#
# slots=True neden kullanıldı?
# - Bu sınıfın "serbest yapıda büyüyen" bir sınıf değil, belirli alanlara sahip bir veri modeli olduğunu vurgular.
# - Rastgele yeni attribute eklenmesini zorlaştırır.
# - Daha kontrollü bir veri modeli hissi verir.
@dataclass(slots=True)
class TokenizerMetrics:
    """
    Tek bir tokenizer için hesaplanan temel metrikleri taşır.

    Bu sınıf, tokenizer çıktısından türetilen sayısal bilgileri saklar.
    Token listesi burada tutulmaz; yalnızca o token listesi üzerinden
    hesaplanan ölçümler tutulur.

    Attributes:
        token_count:
            Toplam token sayısı.
        unique_token_count:
            Tekil (benzersiz) token sayısı.
        unique_ratio:
            Tekil token sayısının toplam token sayısına oranı.
        average_token_length:
            Token uzunluklarının ortalaması.
        min_token_length:
            En kısa token'ın uzunluğu.
        max_token_length:
            En uzun token'ın uzunluğu.
        avg_chars_per_token:
            Toplam karakter sayısının token sayısına bölünmesiyle elde edilen ortalama karakter sayısı.
        unknown_count:
            Bilinemeyen token sayısı (örneğin unk token'lar).
        unknown_rate:
            Bilinemeyen token oranı (unknown_count / token_count).
        latency_seconds:
            Tokenize işleminin ne kadar sürdüğü (saniye cinsinden).
        latency_per_token:
            Token başına düşen ortalama süre (saniye cinsinden).
        efficiency_score:
            Basit bir verimlilik göstergesi (genellikle: token_count / latency_seconds).
        compression_ratio:
            Metin uzunluğunun token sayısına bölünmesiyle hesaplanan oran.
            Basit bir verimlilik göstergesi olarak kullanılabilir.
        top_tokens:
            En sık kullanılan token'ların listesi (token, frekans).
        token_length_distribution:
            Token uzunluklarının dağılımı (uzunluk -> frekans).
        reconstructed_text:
            Tokenizer'ın ürettiği token'lar üzerinden yeniden oluşturulan metin (varsa).
        reconstruction_match:
            Yeniden oluşturulan metin orijinal metinle ne kadar benzer? (örneğin tamamen aynı mı?)
    """

    # Tokenizer tarafından üretilen toplam token sayısı
    token_count: int

    # Tekrar etmeyen, benzersiz token sayısı
    unique_token_count: int

    # Tekil token oranı (unique_token_count / token_count)
    unique_ratio: float

    # Tüm token uzunluklarının ortalaması
    average_token_length: float

    # En kısa token'ın karakter uzunluğu
    min_token_length: int

    # En uzun token'ın karakter uzunluğu
    max_token_length: int

    # Ortalama karakter sayısı / token sayısı
    avg_chars_per_token: float

    # bilinemeyen token sayısı (örneğin unk token'lar)
    unknown_count: int

    # bilinemeyen token oranı (unknown_count / token_count)
    unknown_rate: float

    # tokenize işleminin ne kadar sürdüğü (saniye cinsinden)
    latency_seconds: float

    # token başına düşen ortalama süre (saniye cinsinden)
    latency_per_token: float

    # basit bir verimlilik göstergesi
    # genellikle: token_count / latency_seconds
    efficiency_score: float

    # Basit sıkıştırma/verimlilik oranı
    # Genellikle: len(text) / token_count
    compression_ratio: float

    # En sık kullanılan token'ların listesi (token, frekans)
    top_tokens: list[tuple[str, int]] = field(default_factory=list)

    # Token uzunluklarının dağılımı (uzunluk -> frekans)
    token_length_distribution: dict[int, int] = field(default_factory=dict)

    # Tokenizer'ın ürettiği token'lar üzerinden yeniden oluşturulan metin (varsa)
    reconstructed_text: str | None = None

    # Yeniden oluşturulan metin orijinal metinle ne kadar benzer?
    reconstruction_match: bool | None = None


# ============================================================
# TokenizerEvaluation
# ============================================================
# Bu sınıfın görevi:
# Tek bir tokenizer'ın çalıştırılması sonucunda elde edilen "tam sonucu" saklamaktır.
#
# Bu sınıf neden gerekli?
# Çünkü compare sistemi içinde önce her tokenizer bağımsız olarak değerlendirilir.
# Örneğin:
# - word tokenizer sonucu
# - char tokenizer sonucu
# - byte tokenizer sonucu
# - byte_bpe tokenizer sonucu
#
# Her biri için şu bilgiler olmalıdır:
# - tokenizer adı
# - üretilen token listesi
# - bu token listesi üzerinden hesaplanan metrikler
#
# Eğer bu bilgiler dağınık şekilde dict ile tutulsaydı:
# - kod daha az okunur olurdu
# - tip güvenliği azalırdı
# - test etmek zorlaşırdı
# - formatter ve manager tarafı karışırdı
#
# Bu nedenle her tokenizer sonucu için özel bir model tanımlanmıştır.
@dataclass(slots=True)
class TokenizerEvaluation:
    """
    Tek bir tokenizer'ın değerlendirme sonucunu temsil eder.

    Bu sınıf, compare pipeline içinde bir tokenizer çalıştırıldığında
    oluşan ana sonucu saklar.

    Attributes:
        name:
            Tokenizer'ın görünen adı.
        tokens:
            Tokenizer tarafından üretilen token listesi.
        metrics:
            Bu tokenizer için hesaplanan sayısal metrikler.
    """

    # Tokenizer'ın kullanıcıya gösterilecek kısa adı
    # Örnek: "word", "char", "byte", "byte_bpe"
    name: str

    # Tokenizer tarafından üretilen token listesi
    tokens: list[str]

    # Token listesi üzerinden hesaplanan metrikler
    metrics: TokenizerMetrics


# ============================================================
# PairwiseComparison
# ============================================================
# Bu sınıfın görevi:
# İki tokenizer arasındaki ortak ve farklı token analizini tutmaktır.
#
# Buradaki fikir şu:
# Tek başına token sayısını görmek bazen yeterli değildir.
# Asıl ilginç olan şey tokenizer'ların "nasıl farklı davrandığıdır".
#
# Örnek:
# word tokenizer:
#   ["Hello", "world!", "Tokenization", "is", "fun."]
#
# char tokenizer:
#   ["H", "e", "l", "l", "o", " ", "w", ...]
#
# Bu iki tokenizer arasında:
# - ortak token var mı?
# - sadece solda olanlar neler?
# - sadece sağda olanlar neler?
#
# İşte bu tarz soruların cevabı PairwiseComparison içinde tutulur.
#
# field(default_factory=list) neden kullanıldı?
# Çünkü Python'da mutable default değerleri doğrudan [] şeklinde vermek risklidir.
# Örneğin:
#   common_tokens: list[str] = []
# yazmak sağlıklı değildir.
#
# Bunun yerine:
#   field(default_factory=list)
# yazarsak her nesne oluşturulduğunda yeni bir boş liste üretilir.
@dataclass(slots=True)
class PairwiseComparison:
    """
    İki tokenizer arasındaki ortak ve farklı token analizini tutar.

    Attributes:
        left_name:
            Sol taraftaki tokenizer'ın adı.
        right_name:
            Sağ taraftaki tokenizer'ın adı.
        common_tokens:
            Her iki tokenizer'da da bulunan ortak token'lar.
        unique_to_left:
            Sadece sol tokenizer'da bulunan token'lar.
        unique_to_right:
            Sadece sağ tokenizer'da bulunan token'lar.
        overlap_ratio:
            İki tokenizer arasındaki ortak token oranı 
            (örneğin: len(common_tokens) / max(len(left_tokens), len(right_tokens))).
    """

    # Karşılaştırmanın sol tarafında yer alan tokenizer'ın adı
    left_name: str

    # Karşılaştırmanın sağ tarafında yer alan tokenizer'ın adı
    right_name: str

    # Her iki tokenizer'ın token kümelerinde ortak bulunan token'lar
    common_tokens: list[str] = field(default_factory=list)

    # Sadece sol tokenizer'da bulunan token'lar
    unique_to_left: list[str] = field(default_factory=list)

    # Sadece sağ tokenizer'da bulunan token'lar
    unique_to_right: list[str] = field(default_factory=list)

    # İki tokenizer arasındaki ortak token oranı 
    # (örneğin: len(common_tokens) / max(len(left_tokens), len(right_tokens)))
    overlap_ratio: float | None = None


# ============================================================
# ComparisonResult
# ============================================================
# Bu sınıfın görevi:
# Tüm compare pipeline sonucunu TEK bir nesne içinde toplamaktır.
#
# CompareManager.compare(...) çalıştığında elimizde sadece tek bir tokenizer sonucu olmayacaktır. 
# Birden fazla tokenizer için evaluation sonucu ve ayrıca onların pairwise comparison bilgileri de olacaktır.
#
# Bu yüzden ana bir kapsayıcı modele ihtiyaç duyulmaktadır.
#
# Bu sınıf şunu sağlamaktadır:
# - karşılaştırmada hangi metin kullanıldı?
# - her tokenizer ne üretti?
# - tokenizer'lar birbirinden nasıl ayrıştı?
#
# Böylece CompareManager sonucu tek bir güçlü veri modeli olarak döndürebilir.
# Sonrasında bu veri:
# - terminal çıktısına dönüştürülebilir
# - testlerde doğrulanabilir
# - JSON'a çevrilebilir
# - report formatter'a gönderilebilir
@dataclass(slots=True)
class ComparisonResult:
    """
    Tüm tokenizer karşılaştırma sonucunu taşıyan ana modeldir.

    Bu sınıf compare pipeline'ının sonunda dönecek ana veri yapısıdır.
    Tek tek tokenizer sonuçlarını ve tokenizer çiftleri arasındaki karşılaştırmaları bir arada tutar.

    Attributes:
        source_text:
            Karşılaştırmada kullanılan orijinal metin.
        evaluations:
            Her tokenizer için oluşturulan evaluation sonuçları.
        pairwise_comparisons:
            Tokenizer çiftleri arasındaki ortak/farklı token analizleri.
    """

    # Tokenizer'ların üzerinde çalıştığı orijinal metin
    source_text: str

    # Her tokenizer için oluşturulmuş değerlendirme sonuçları
    # Örnek:
    # [
    #   TokenizerEvaluation(name="word", ...),
    #   TokenizerEvaluation(name="char", ...),
    #   TokenizerEvaluation(name="byte_bpe", ...),
    # ]
    evaluations: list[TokenizerEvaluation]

    # Tokenizer çiftleri arasındaki karşılaştırma sonuçları
    # Örnek:
    # [
    #   PairwiseComparison(left_name="word", right_name="char", ...),
    #   PairwiseComparison(left_name="word", right_name="byte_bpe", ...),
    # ]
    pairwise_comparisons: list[PairwiseComparison]