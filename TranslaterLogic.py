
nVowels =26
# 46
consonants = []
consonantsUni = []

# 26
vowels = []
vowelsUni = []
vowelModifiersUni = []

# 6
specialConsonants = []
specialConsonantsUni = []
# 2
specialCharUni = []
specialChar = []


def initializeVar():

    vowelsUni.append("ඌ")
    vowels.append("oo")
    vowelModifiersUni.append("ූ")

    vowelsUni.append("ඕ")
    vowels.append("o\\)")
    vowelModifiersUni.append("ෝ")
    vowelsUni.append("ඕ")
    vowels.append("oe")
    vowelModifiersUni.append("ෝ")
    vowelsUni.append("ආ")
    vowels.append("aa")
    vowelModifiersUni.append("ා")
    vowelsUni.append("ආ")
    vowels.append("a\\)")
    vowelModifiersUni.append("ා")
    vowelsUni.append("ඈ")
    vowels.append("Aa")
    vowelModifiersUni.append("ෑ")
    vowelsUni.append("ඈ")
    vowels.append("A\\)")
    vowelModifiersUni.append("ෑ")
    vowelsUni.append("ඈ")
    vowels.append("ae")
    vowelModifiersUni.append("ෑ")
    vowelsUni.append("ඊ")
    vowels.append("ii")
    vowelModifiersUni.append("ී")
    vowelsUni.append("ඊ")
    vowels.append("i\\)")
    vowelModifiersUni.append("ී")
    vowelsUni.append("ඊ")
    vowels.append("ie")
    vowelModifiersUni.append("ී")
    vowelsUni.append("ඊ")
    vowels.append("ee")
    vowelModifiersUni.append("ී")
    vowelsUni.append("ඒ")
    vowels.append("ea")
    vowelModifiersUni.append("ේ")
    vowelsUni.append("ඒ")
    vowels.append("e\\)")
    vowelModifiersUni.append("ේ")
    vowelsUni.append("ඒ")
    vowels.append("ei")
    vowelModifiersUni.append("ේ")
    vowelsUni.append("ඌ")
    vowels.append("uu")
    vowelModifiersUni.append("ූ")
    vowelsUni.append("ඌ")
    vowels.append("u\\)")
    vowelModifiersUni.append("ූ")

    vowelsUni.append("ඖ")
    vowels.append("au")
    vowelModifiersUni.append("ෞ")

    vowelsUni.append("ඇ")
    vowels.append("\\a")
    vowelModifiersUni.append("ැ")

    vowelsUni.append("අ")
    vowels.append("a")
    vowelModifiersUni.append("")

    vowelsUni.append("ඇ")
    vowels.append("A")
    vowelModifiersUni.append("ැ")
    vowelsUni.append("ඉ")
    vowels.append("i")
    vowelModifiersUni.append("ි")
    vowelsUni.append("එ")
    vowels.append("e")
    vowelModifiersUni.append("ෙ")
    vowelsUni.append("උ")
    vowels.append("u")
    vowelModifiersUni.append("ු")
    vowelsUni.append("ඔ")
    vowels.append("o")
    vowelModifiersUni.append("ො")
    vowelsUni.append("ඓ")
    vowels.append("I")
    vowelModifiersUni.append("ෛ")

    specialConsonantsUni.append("ං")
    specialConsonants.append("\\n")

    specialConsonantsUni.append("ඃ")
    specialConsonants.append("\\h")
    specialConsonantsUni.append("ඞ")
    specialConsonants.append("\\N")
    specialConsonantsUni.append("ඍ")
    specialConsonants.append("\\R")
    # special characher Repaya
    specialConsonantsUni.append("ර්"+"\u200D")
    specialConsonants.append("R")
    specialConsonantsUni.append("ර්"+"\u200D")
    specialConsonants.append("\\r")

    consonantsUni.append("ඬ")
    consonants.append("nnd")

    consonantsUni.append("ඳ")
    consonants.append("nndh")

    consonantsUni.append("ඟ")
    consonants.append("nng")

    consonantsUni.append("ත")
    consonants.append("th")

    consonantsUni.append("ධ")
    consonants.append("dh")
    consonantsUni.append("ඝ")
    consonants.append("gh")
    consonantsUni.append("ච")
    consonants.append("ch")
    consonantsUni.append("ඵ")
    consonants.append("ph")
    consonantsUni.append("භ")
    consonants.append("bh")
    consonantsUni.append("ඣ")
    consonants.append("jh")
    consonantsUni.append("ෂ")
    consonants.append("sh")
    consonantsUni.append("ඥ")
    consonants.append("GN")
    consonantsUni.append("ඤ")
    consonants.append("KN")
    consonantsUni.append("ළු")
    consonants.append("Lu")
    consonantsUni.append("ඛ")
    consonants.append("kh")
    consonantsUni.append("ඨ")
    consonants.append("Th")
    consonantsUni.append("ඪ")
    consonants.append("Dh")
    consonantsUni.append("ශ")
    consonants.append("S")
    consonantsUni.append("ද")
    consonants.append("d")
    consonantsUni.append("ච")
    consonants.append("c")
    consonantsUni.append("ත")
    consonants.append("th")
    consonantsUni.append("ට")
    consonants.append("t")
    consonantsUni.append("ක")
    consonants.append("k")
    consonantsUni.append("ඩ")
    consonants.append("D")
    consonantsUni.append("න")
    consonants.append("n")
    consonantsUni.append("ප")
    consonants.append("p")
    consonantsUni.append("බ")
    consonants.append("b")
    consonantsUni.append("ම")
    consonants.append("m")
    consonantsUni.append("‍ය")
    consonants.append("\\u005C" + "y")
    consonantsUni.append("‍ය")
    consonants.append("Y")
    consonantsUni.append("ය")
    consonants.append("y")
    consonantsUni.append("ජ")
    consonants.append("j")
    consonantsUni.append("ල")
    consonants.append("l")
    consonantsUni.append("ව")
    consonants.append("v")
    consonantsUni.append("ව")
    consonants.append("w")
    consonantsUni.append("ස")
    consonants.append("s")
    consonantsUni.append("හ")
    consonants.append("h")
    consonantsUni.append("ණ")
    consonants.append("N")
    consonantsUni.append("ළ")
    consonants.append("L")
    consonantsUni.append("ඛ")
    consonants.append("K")
    consonantsUni.append("ඝ")
    consonants.append("G")
    consonantsUni.append("ඵ")
    consonants.append("P")
    consonantsUni.append("ඹ")
    consonants.append("B")
    consonantsUni.append("ෆ")
    consonants.append("f")
    consonantsUni.append("ග")
    consonants.append("g")
    # last because we need to ommit this in dealing with Rakaransha
    consonantsUni.append("ර")
    consonants.append("r")
    specialCharUni.append("ෲ")
    specialChar.append("ruu")
    specialCharUni.append("ෘ")
    specialChar.append("ru")
    # specialCharUni[2]="්‍ර" specialChar[2]="ra"

initializeVar()

def convertText(text):
    s=""
    r=""
    v=""
    # text = document.txtBox.box1.value;
    # special consonents
    for i in range(len(specialConsonants)):
        text = text.replace(specialConsonants[i], specialConsonantsUni[i])
# consonents + special Chars
    for i in range(len(specialCharUni)):
        for j in range(len(consonants)):
            s = consonants[j] + specialChar[i]
            v = consonantsUni[j] + specialCharUni[i]
            # r = new RegExp(s, "g")
            r = s.replace(s+"/G", "")
            text = text.replace(r, v)
        
# consonants + Rakaransha + vowel modifiers
    for j in range(len(consonants)):
        for i in range(len(vowels)):
            s = consonants[j] + "r" + vowels[i]
            v = consonantsUni[j] + "්‍ර" + vowelModifiersUni[i]
            # r = new RegExp(s, "g")
            r = s.replace(s+"/G", "")
            text = text.replace(r, v)
        s = consonants[j] + "r"
        v = consonantsUni[j] + "්‍ර"
        # r = new RegExp(s, "g")
        r = s.replace(s+"/G", "")
        text = text.replace(r, v)
# consonents + vowel modifiers
    for i in range(len(consonants)):
        for j in range(nVowels):
            s = consonants[i] + vowels[j]
            v = consonantsUni[i] + vowelModifiersUni[j]
            # r = new RegExp(s, "g")
            r = s.replace(s+"/G", "")
            text = text.replace(r, v)
        
# consonents + HAL
    for i in range(len(consonants)):
        r = consonants[i].replace(consonants[i]+"/G", "")
        text = text.replace(r, consonantsUni[i] + "්")
# vowels
    for i in range(len(vowels)):
        # r = new RegExp(vowels[i], "g")
        r = vowels[i].replace(vowels[i]+"/G", "")
        text = text.replace(r, vowelsUni[i])

    return text

