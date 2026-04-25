import re 
from collections import defaultdict

# Metni parçalara bölmek için regex pattern
# [a-zA-Z...]+ → kelimeler
# \d+          → rakamlar
# [^\s\w]+     → noktalama işaretleri
# \s+          → boşluklar
SIMPLE_PATTERN = r"[a-zA-ZğüşıöçĞÜŞİÖÇə]+|\d+|[^\s\w]+|\s+"

class RegexBPETokenizer:

    def __init__(self, pattern: str = SIMPLE_PATTERN):
        # Kullanıcı pattern vermezse SIMPLE_PATTERN kullanılır
        self.pattern = pattern
        # Sözlük: id → bytes {0: b'a', 256: b'sa', ...}
        self.vocab = {}
        # Öğrenilmiş birleştirme kuralları: (id1,id2) → yeni_id
        self.merges = {}
        # Pattern bir kez compile edilir — hızlı kullanım için
        self._compiled = re.compile(pattern)

    def _pre_tokenize(self, text: str) -> list[str]:
        # Metni regex ile anlamlı parçalara böler
        # "Salam, dünya!" → ["Salam", ",", " dünya", "!"]
        # BPE'den ÖNCE çalışır — kelime sınırlarını korur
        return self._compiled.findall(text)

    def _text_to_ids(self, text: str) -> list[int]:
        # Metin parçasını UTF-8 byte id'lerine çevirir
        # "sə" → [115, 201, 99]
        # list() — çünkü bytes değiştirilemez, list değiştirilebilir
        return list(text.encode("utf-8"))

    def _get_stats(self, ids: list[int]) -> dict:
        # Tüm komşu çiftleri sayar
        # ids    = [115,  97, 108,  97, 109]
        # ids[1:]= [ 97, 108,  97, 109]
        # zip    → (115,97), (97,108), (108,97), (97,109)
        # defaultdict(int) — key yoksa otomatik 0 verir
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1  # her çift görüldüğünde sayıyı 1 artır
        return counts

    def _merge(self, ids: list[int], pair: tuple, new_id: int) -> list[int]:
        # ids'de verilen çifti bulur ve new_id ile değiştirir
        # ids=[115,97,108], pair=(115,97), new_id=256
        # → [256, 108]
        result = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                # Çift bulundu → new_id ekle
                # i += 2 — çünkü 2 elementi 1 ile değiştirdik
                result.append(new_id)
                i += 2
            else:
                # Çift bulunamadı → elementi olduğu gibi ekle
                result.append(ids[i])
                i += 1
        return result
    
    def _build_vocab(self) -> None:
    # 0-255 arası temel byte tokenları oluştur
    # bytes([115]) → b's', bytes([97]) → b'a' gibi
    # Yani 256 temel token her zaman sözlükte vardır
        self.vocab = {i: bytes([i]) for i in range(256)}
    
        # Öğrenilmiş birleştirme kurallarını sözlüğe ekle
        # merges = {(115,97): 256, (256,108): 257, ...}
        for (p0, p1), idx in self.merges.items():
            # p0=115 → b's'
            # p1=97  → b'a'
            # vocab[256] = b's' + b'a' = b'sa'
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
    
    def train(self, text: str, vocab_size: int) -> None:
    # vocab_size 256'dan küçükse hata ver
    # Çünkü 0-255 byte tokenları her zaman olmalıdır
        if vocab_size < 256:
            raise ValueError("vocab_size minimum 256 olmalıdır.")
    
    # Kaç tane yeni birleştirme öğreneceğiz?
    # vocab_size=270 → 270-256=14 yeni birleştirme
        num_merges = vocab_size - 256
    
    # Metni regex ile parçalara böl
    # "salam dünya" → ["salam", " dünya"]
        chunks = self._pre_tokenize(text)
    
    # Her parçayı byte id'lerine çevir
    # ["salam", " dünya"] → [[115,97,108,97,109], [32,100,...]]
        ids_list = [self._text_to_ids(chunk) for chunk in chunks]
    
    # Her train çağrısında eski kuralları sıfırla
        self.merges = {}

    # num_merges kadar tekrarla — her turda 1 yeni kural öğren
        for i in range(num_merges):
        
        # Tüm parçalardaki çift sayılarını topla
            total_stats = defaultdict(int)
            for ids in ids_list:
            # Her parçanın istatistiğini al
            # {(115,97):1, (97,108):1, ...}
                for pair, count in self._get_stats(ids).items():
                # Tüm parçaların sayılarını birleştir
                    total_stats[pair] += count
        
        # Birleştirilecek çift kalmadıysa dur
            if not total_stats:
                break
        
        # En çok tekrarlanan çifti bul
        # total_stats = {(115,97):3, (97,108):2}
        # best_pair   = (115,97) ← en çok bu!
            best_pair = max(total_stats, key=lambda p: total_stats[p])
        
        # Yeni token id'si belirle
        # i=0 → 256, i=1 → 257, i=2 → 258 ...
            new_id = 256 + i
        
        # Tüm parçalarda bu çifti yeni id ile değiştir
        # [[115,97,108]] → [[256,108]]
            ids_list = [self._merge(ids, best_pair, new_id) for ids in ids_list]
        
        # Kuralı kaydet
        # merges[(115,97)] = 256
            self.merges[best_pair] = new_id
    
    # Tüm birleştirmeler öğrenildi
    # Şimdi decode için sözlüğü oluştur
        self._build_vocab()

    def encode(self, text: str) -> list[int]:
    # Train edilmeden encode çağrılırsa hata ver
    # merges ve vocab boşsa → train() çağrılmamış demektir
        if not self.merges and not self.vocab:
            raise RuntimeError("Önce train() kullanılmalı")
    
    # Metni regex ile parçalara böl
    # "salam dünya" → ["salam", " dünya"]
        chunks = self._pre_tokenize(text)
    
    # Tüm parçaların id'lerini bir yerde toplamak için
        all_ids = []

        for chunk in chunks:
        # Her parçayı byte id'lerine çevir
        # "salam" → [115,97,108,97,109]
            ids = self._text_to_ids(chunk)

        # Öğrenilmiş tüm kuralları sırayla uygula
        # merges = {(115,97):256, (256,108):257, ...}
            for pair, new_id in self.merges.items():
            # Her kuralı uygula — çifti yeni id ile değiştir
            # [115,97,108] → [256,108] → [257] ...
                ids = self._merge(ids, pair, new_id)
        
        # Bu parçanın id'lerini genel listeye ekle
        # extend — listeyi listeye birleştirir
        # all_ids=[258,109], ids=[32,100] → [258,109,32,100]
            all_ids.extend(ids)
    
    # Tüm parçaların id'lerini döndür
        return all_ids
    
    def decode(self, ids: list[int]) -> str:
    # Train edilmeden decode çağrılırsa hata ver
    # vocab boşsa → train() çağrılmamış demektir
        if not self.vocab:
            raise RuntimeError("Önce train() kullanılmalı")
    
    # Her id'yi vocab'dan bytes'a çevir
    # ids=[258,109] → [b'sala', b'm']
    # vocab[258]=b'sala', vocab[109]=b'm'
        byte_parts = [self.vocab[i] for i in ids]
    
    # Tüm bytes parçalarını birleştir ve metne çevir
    # b"".join → [b'sala', b'm'] → b'salam'
    # .decode("utf-8") → b'salam' → "salam"
    # errors="replace" → hatalı byte gelirse "?" ile değiştir
        return b"".join(byte_parts).decode("utf-8", errors="replace")
    
    def get_vocab_size(self) -> int:
    # Sözlüğün kaç token içerdiğini döndürür
    # vocab = {0:b'a', 1:b'b', ..., 256:b'sa', ...}
    # len(vocab) → toplam token sayısı
        return len(self.vocab)

    def get_merges_count(self) -> int:
    # Kaç tane birleştirme kuralı öğrenildiğini döndürür
    # merges = {(115,97):256, (256,108):257, ...}
    # len(merges) → kural sayısı
        return len(self.merges)
    