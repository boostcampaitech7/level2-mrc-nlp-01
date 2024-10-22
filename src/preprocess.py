# https://unicode-explorer.com/blocks
ranges = {
    "basic_latin": "\u0000-\u007F",
    "latin1_supplement": "\u0080-\u00FF",
    "latin_extended_a": "\u0100-\u017F",
    "latin_extended_b": "\u0180-\u024F",
    "ipa_extensions": "\u0250-\u02AF",
    "spacing_modifiers": "\u02B0-\u02FF",
    "combining_diacritical_marks": "\u0300-\u036F",
    "greek_coptic": "\u0370-\u03FF",
    "cyrillic": "\u0400-\u04FF",
    "cyrillic_supplement": "\u0500-\u052F",
    "armenian": "\u0530-\u058F",
    "hebrew": "\u0590-\u05FF",
    "arabic": "\u0600-\u06FF",
    "syriac": "\u0700-\u074F",
    "arabic_supplement": "\u0750-\u077F",
    "thaana": "\u0780-\u07BF",
    "nko": "\u07C0-\u07FF",
    "samaritan": "\u0800-\u083F",
    "mandaic": "\u0840-\u085F",
    "arabic_extended_a": "\u08A0-\u08FF",
    "devanagari": "\u0900-\u097F",
    "bengali": "\u0980-\u09FF",
    "gurmukhi": "\u0A00-\u0A7F",
    "gujarati": "\u0A80-\u0AFF",
    "oriya": "\u0B00-\u0B7F",
    "tamil": "\u0B80-\u0BFF",
    "telugu": "\u0C00-\u0C7F",
    "kannada": "\u0C80-\u0CFF",
    "malayalam": "\u0D00-\u0D7F",
    "sinhala": "\u0D80-\u0DFF",
    "thai": "\u0E00-\u0E7F",
    "lao": "\u0E80-\u0EFF",
    "tibetan": "\u0F00-\u0FFF",
    "myanmar": "\u1000-\u109F",
    "georgian": "\u10A0-\u10FF",
    "hangul_jamo": "\u1100-\u11FF",
    "ethiopic": "\u1200-\u137F",
    "ethiopic_supplement": "\u1380-\u139F",
    "cherokee": "\u13A0-\u13FF",
    "unified_canadian_aboriginal_syllabics": "\u1400-\u167F",
    "ogham": "\u1680-\u169F",
    "runic": "\u16A0-\u16FF",
    "tagalog": "\u1700-\u171F",
    "hanunoo": "\u1720-\u173F",
    "buhid": "\u1740-\u175F",
    "tagbanwa": "\u1760-\u177F",
    "khmer": "\u1780-\u17FF",
    "mongolian": "\u1800-\u18AF",
    "unified_canadian_aboriginal_syllabics_extended": "\u18B0-\u18FF",
    "limbu": "\u1900-\u194F",
    "tai_le": "\u1950-\u197F",
    "new_tai_lue": "\u1980-\u19DF",
    "khmer_symbols": "\u19E0-\u19FF",
    "buginese": "\u1A00-\u1A1F",
    "tai_tham": "\u1A20-\u1AAF",
    "combining_diacritical_marks_extended": "\u1AB0-\u1AFF",
    "balinese": "\u1B00-\u1B7F",
    "sundanese": "\u1B80-\u1BBF",
    "batak": "\u1BC0-\u1BFF",
    "lepcha": "\u1C00-\u1C4F",
    "ol_chiki": "\u1C50-\u1C7F",
    "cyrillic_extended_c": "\u1C80-\u1C8F",
    "georgian_extended": "\u1C90-\u1CBF",
    "sundanese_supplement": "\u1CC0-\u1CCF",
    "vedic_extensions": "\u1CD0-\u1CFF",
    "phonetic_extensions": "\u1D00-\u1D7F",
    "phonetic_extensions_supplement": "\u1D80-\u1DBF",
    "combining_diacritical_marks_supplement": "\u1DC0-\u1DFF",
    "latin_extended_additional": "\u1E00-\u1EFF",
    "greek_extended": "\u1F00-\u1FFF",
    "general_punctuation": "\u2000-\u206F",
    "superscripts_and_subscripts": "\u2070-\u209F",
    "currency_symbols": "\u20A0-\u20CF",
    "combining_diacritical_marks_for_symbols": "\u20D0-\u20FF",
    "letterlike_symbols": "\u2100-\u214F",
    "number_forms": "\u2150-\u218F",
    "arrows": "\u2190-\u21FF",
    "mathematical_operators": "\u2200-\u22FF",
    "miscellaneous_technical": "\u2300-\u23FF",
    "control_pictures": "\u2400-\u243F",
    "optical_character_recognition": "\u2440-\u245F",
    "enclosed_alphanumerics": "\u2460-\u24FF",
    "box_drawing": "\u2500-\u257F",
    "block_elements": "\u2580-\u259F",
    "geometric_shapes": "\u25A0-\u25FF",
    "miscellaneous_symbols": "\u2600-\u26FF",
    "dingbats": "\u2700-\u27BF",
    "miscellaneous_mathematical_symbols_a": "\u27C0-\u27EF",
    "supplemental_arrows_a": "\u27F0-\u27FF",
    "braille_patterns": "\u2800-\u28FF",
    "supplemental_arrows_b": "\u2900-\u297F",
    "miscellaneous_mathematical_symbols_b": "\u2980-\u29FF",
    "supplemental_mathematical_operators": "\u2A00-\u2AFF",
    "miscellaneous_symbols_and_arrows": "\u2B00-\u2BFF",
    "glagolitic": "\u2C00-\u2C5F",
    "latin_extended_c": "\u2C60-\u2C7F",
    "coptic": "\u2C80-\u2CFF",
    "georgian_supplement": "\u2D00-\u2D2F",
    "tifinagh": "\u2D30-\u2D7F",
    "ethiopic_extended": "\u2D80-\u2DDF",
    "cyrillic_extended_a": "\u2DE0-\u2DFF",
    "supplemental_punctuation": "\u2E00-\u2E7F",
    "cjk_radicals_supplement": "\u2E80-\u2EFF",
    "kangxi_radicals": "\u2F00-\u2FDF",
    "ideographic_description_characters": "\u2FF0-\u2FFF",
    "cjk_symbols_and_punctuation": "\u3000-\u303F",
    "hiragana": "\u3040-\u309F",
    "katakana": "\u30A0-\u30FF",
    "bopomofo": "\u3100-\u312F",
    "hangul_compatibility_jamo": "\u3130-\u318F",
    "kanbun": "\u3190-\u319F",
    "bopomofo_extended": "\u31A0-\u31BF",
    "cjk_strokes": "\u31C0-\u31EF",
    "katakana_phonetic_extensions": "\u31F0-\u31FF",
    "enclosed_cjk_letters_and_months": "\u3200-\u32FF",
    "cjk_compatibility": "\u3300-\u33FF",
    "cjk_unified_ideographs_extension_a": "\u3400-\u4DBF",
    "yijing_hexagram_symbols": "\u4DC0-\u4DFF",
    "cjk_unified_ideographs": "\u4E00-\u9FFF",
    "yi_syllables": "\uA000-\uA48F",
    "yi_radicals": "\uA490-\uA4CF",
    "lisu": "\uA4D0-\uA4FF",
    "vai": "\uA500-\uA63F",
    "cyrillic_extended_b": "\uA640-\uA69F",
    "bamum": "\uA6A0-\uA6FF",
    "modifier_tone_letters": "\uA700-\uA71F",
    "latin_extended_d": "\uA720-\uA7FF",
    "syloti_nagri": "\uA800-\uA82F",
    "common_indic_number_forms": "\uA830-\uA83F",
    "phags_pa": "\uA840-\uA87F",
    "saurashtra": "\uA880-\uA8DF",
    "devanagari_extended": "\uA8E0-\uA8FF",
    "kayah_li": "\uA900-\uA92F",
    "rejang": "\uA930-\uA95F",
    "hangul_jamo_extended_a": "\uA960-\uA97F",
    "javanese": "\uA980-\uA9DF",
    "myanmar_extended_b": "\uA9E0-\uA9FF",
    "cham": "\uAA00-\uAA5F",
    "myanmar_extended_a": "\uAA60-\uAA7F",
    "tai_viet": "\uAA80-\uAADF",
    "meetei_mayek_extensions": "\uAAE0-\uAAFF",
    "ethiopic_extended_a": "\uAB00-\uAB2F",
    "latin_extended_e": "\uAB30-\uAB6F",
    "cherokee_supplement": "\uAB70-\uABBF",
    "meetei_mayek": "\uABC0-\uABFF",
    "hangul_syllables": "\uAC00-\uD7AF",
    "hangul_jamo_extended_b": "\uD7B0-\uD7FF",
    "high_surrogates": "\uD800-\uDB7F",
    "high_private_use_surrogates": "\uDB80-\uDBFF",
    "low_surrogates": "\uDC00-\uDFFF",
    "private_use_area": "\uE000-\uF8FF",
    "cjk_compatibility_ideographs": "\uF900-\uFAFF",
    "alphabetic_presentation_forms": "\uFB00-\uFB4F",
    "arabic_presentation_forms_a": "\uFB50-\uFDFF",
    "variation_selectors": "\uFE00-\uFE0F",
    "vertical_forms": "\uFE10-\uFE1F",
    "combining_half_marks": "\uFE20-\uFE2F",
    "cjk_compatibility_forms": "\uFE30-\uFE4F",
    "small_form_variants": "\uFE50-\uFE6F",
    "arabic_presentation_forms_b": "\uFE70-\uFEFF",
    "halfwidth_and_fullwidth_forms": "\uFF00-\uFFEF",
    "specials": "\uFFF0-\uFFFF",
    "linear_b_syllabary": "\U00010000-\U0001007F",
    "linear_b_ideograms": "\U00010080-\U000100FF",
    "aegean_numbers": "\U00010100-\U0001013F",
    "ancient_greek_numbers": "\U00010140-\U0001018F",
    "ancient_symbols": "\U00010190-\U000101CF",
    "phaistos_disc": "\U000101D0-\U000101FF",
    "lycian": "\U00010280-\U0001029F",
    "carian": "\U000102A0-\U000102DF",
    "coptic_epact_numbers": "\U000102E0-\U000102FF",
    "old_italic": "\U00010300-\U0001032F",
    "gothic": "\U00010330-\U0001034F",
    "old_permic": "\U00010350-\U0001037F",
    "ugaritic": "\U00010380-\U0001039F",
    "old_persian": "\U000103A0-\U000103DF",
    "deseret": "\U00010400-\U0001044F",
    "shavian": "\U00010450-\U0001047F",
    "osmanya": "\U00010480-\U000104AF",
    "osage": "\U000104B0-\U000104FF",
    "elbasan": "\U00010500-\U0001052F",
    "caucasian_albanian": "\U00010530-\U0001056F",
    "linear_a": "\U00010600-\U0001077F",
    "linear_b_ideograms": "\U00010800-\U0001083F",
    "linear_b_syllabary": "\U00010840-\U00010855",
    "cypriot_syllabary": "\U00010900-\U0001091F",
    "imperial_aramaic": "\U00010840-\U00010855",
    "palmyrene": "\U00010860-\U0001087F",
    "nabataean": "\U00010880-\U0001089F",
    "hatran": "\U000108E0-\U000108FF",
    "phoenician": "\U00010900-\U0001091F",
    "lydian": "\U00010920-\U0001093F",
    "meroitic_hieroglyphs": "\U00010980-\U0001099F",
    "meroitic_cursive": "\U000109A0-\U000109FF",
    "kharoshthi": "\U00010A00-\U00010A5F",
    "old_south_arabian": "\U00010A60-\U00010A7F",
    "old_north_arabian": "\U00010A80-\U00010A9F",
    "manichaean": "\U00010AC0-\U00010AFF",
    "avestan": "\U00010B00-\U00010B3F",
    "inscriptional_parthian": "\U00010B40-\U00010B5F",
    "inscriptional_pahlavi": "\U00010B60-\U00010B7F",
    "psalter_pahlavi": "\U00010B80-\U00010BAF",
    "old_turkic": "\U00010C00-\U00010C4F",
    "old_hungarian": "\U00010C80-\U00010CFF",
    "hanifi_rohingya": "\U00010D00-\U00010D3F",
    "rumi_numeral_symbols": "\U00010E60-\U00010E7F",
    "yezidi": "\U00010E80-\U00010EBF",
    "old_sogdian": "\U00010F00-\U00010F2F",
    "sogdian": "\U00010F30-\U00010F6F",
    "old_uyghur": "\U00010F70-\U00010FAF",
    "chorasmian": "\U00010FB0-\U00010FDF",
    "elymaic": "\U00010FE0-\U00010FFF",
    "brahmi": "\U00011000-\U0001107F",
    "kaithi": "\U00011080-\U000110CF",
    "sora_sompeng": "\U000110D0-\U000110FF",
    "chakma": "\U00011100-\U0001114F",
    "mahajani": "\U00011150-\U0001117F",
    "sharada": "\U00011180-\U000111DF",
    "sinhala_archaic_numbers": "\U000111E0-\U000111FF",
    "khojki": "\U00011200-\U0001124F",
    "multani": "\U00011280-\U000112AF",
    "khudawadi": "\U000112B0-\U000112FF",
    "grantha": "\U00011300-\U0001137F",
    "newa": "\U00011400-\U0001147F",
    "tirhuta": "\U00011480-\U000114DF",
    "siddham": "\U00011580-\U000115FF",
    "modi": "\U00011600-\U0001165F",
    "mongolian_supplement": "\U00011660-\U0001167F",
    "takri": "\U00011680-\U000116CF",
    "ahom": "\U00011700-\U0001173F",
    "dogra": "\U00011800-\U0001184F",
    "warang_citi": "\U000118A0-\U000118FF",
    "dives_akuru": "\U00011900-\U0001195F",
    "nandinagari": "\U000119A0-\U000119FF",
    "zanabazar_square": "\U00011A00-\U00011A4F",
    "soyombo": "\U00011A50-\U00011AAF",
    "pau_cin_hau": "\U00011AC0-\U00011AFF",
    "bhaiksuki": "\U00011C00-\U00011C6F",
    "marchen": "\U00011C70-\U00011CBF",
    "masaram_gondi": "\U00011D00-\U00011D5F",
    "gunjala_gondi": "\U00011D60-\U00011DAF",
    "makasar": "\U00011EE0-\U00011EFF",
    "lisu_supplement": "\U00011FB0-\U00011FBF",
    "tamil_supplement": "\U00011FC0-\U00011FFF",
    "cuneiform": "\U00012000-\U000123FF",
    "cuneiform_numbers_and_punctuation": "\U00012400-\U0001247F",
    "early_dynastic_cuneiform": "\U00012480-\U0001254F",
    "egyptian_hieroglyphs": "\U00013000-\U0001342F",
    "egyptian_hieroglyph_format_controls": "\U00013430-\U0001343F",
    "anatolian_hieroglyphs": "\U00014400-\U0001467F",
    "bamum_supplement": "\U00016800-\U00016A3F",
    "mro": "\U00016A40-\U00016A6F",
    "bassa_vah": "\U00016AD0-\U00016AFF",
    "pahawh_hmong": "\U00016B00-\U00016B8F",
    "medefaidrin": "\U00016E40-\U00016E9F",
    "miao": "\U00016F00-\U00016F9F",
    "ideographic_symbols_and_punctuation": "\U00016FE0-\U00016FFF",
    "tangut": "\U00017000-\U000187FF",
    "tangut_components": "\U00018800-\U00018AFF",
    "khitan_small_script": "\U00018B00-\U00018CFF",
    "tangut_supplement": "\U00018D00-\U00018D8F",
    "kana_supplement": "\U0001B000-\U0001B0FF",
    "kana_extended_a": "\U0001B100-\U0001B12F",
    "small_kana_extension": "\U0001B130-\U0001B16F",
    "nushu": "\U0001B170-\U0001B2FF",
    "duployan": "\U0001BC00-\U0001BC9F",
    "shorthand_format_controls": "\U0001BCA0-\U0001BCAF",
    "byzantine_musical_symbols": "\U0001D000-\U0001D0FF",
    "musical_symbols": "\U0001D100-\U0001D1FF",
    "ancient_greek_musical_notation": "\U0001D200-\U0001D24F",
    "mayan_numerals": "\U0001D2E0-\U0001D2FF",
    "tai_xuan_jing_symbols": "\U0001D300-\U0001D35F",
    "counting_rod_numerals": "\U0001D360-\U0001D37F",
    "mathematical_alphanumeric_symbols": "\U0001D400-\U0001D7FF",
    "sutton_signwriting": "\U0001D800-\U0001DAAF",
    "glagolitic_supplement": "\U0001E000-\U0001E02F",
    "nyiakeng_puachue_hmong": "\U0001E100-\U0001E14F",
    "wancho": "\U0001E2C0-\U0001E2FF",
    "mende_kikakui": "\U0001E800-\U0001E8DF",
    "adlam": "\U0001E900-\U0001E95F",
    "indic_siyaq_numbers": "\U0001EC70-\U0001ECBF",
    "ottoman_siyaq_numbers": "\U0001ED00-\U0001ED4F",
    "arabic_mathematical_alphabetic_symbols": "\U0001EE00-\U0001EEFF",
    "mahjong_tiles": "\U0001F000-\U0001F02F",
    "domino_tiles": "\U0001F030-\U0001F09F",
    "playing_cards": "\U0001F0A0-\U0001F0FF",
    "enclosed_alphanumeric_supplement": "\U0001F100-\U0001F1FF",
    "enclosed_ideographic_supplement": "\U0001F200-\U0001F2FF",
    "miscellaneous_symbols_and_pictographs": "\U0001F300-\U0001F5FF",
    "emoticons": "\U0001F600-\U0001F64F",
    "ornamental_dingbats": "\U0001F650-\U0001F67F",
    "transport_and_map_symbols": "\U0001F680-\U0001F6FF",
    "alchemical_symbols": "\U0001F700-\U0001F77F",
    "geometric_shapes_extended": "\U0001F780-\U0001F7FF",
    "supplemental_arrows_c": "\U0001F800-\U0001F8FF",
    "supplemental_symbols_and_pictographs": "\U0001F900-\U0001F9FF",
    "chess_symbols": "\U0001FA00-\U0001FA6F",
    "symbols_and_pictographs_extended_a": "\U0001FA70-\U0001FAFF",
    "symbols_for_legacy_computing": "\U0001FB00-\U0001FBFF",
    "cjk_unified_ideographs_extension_b": "\U00020000-\U0002A6DF",
    "cjk_unified_ideographs_extension_c": "\U0002A700-\U0002B73F",
    "cjk_unified_ideographs_extension_d": "\U0002B740-\U0002B81F",
    "cjk_unified_ideographs_extension_e": "\U0002B820-\U0002CEAF",
    "cjk_unified_ideographs_extension_f": "\U0002CEB0-\U0002EBEF",
    "cjk_compatibility_ideographs_supplement": "\U0002F800-\U0002FA1F",
    "tags": "\U000E0000-\U000E007F",
    "variation_selectors_supplement": "\U000E0100-\U000E01EF",
    "supplementary_private_use_area_a": "\U000F0000-\U000FFFFF",
    "supplementary_private_use_area_b": "\U00100000-\U0010FFFF",
}

import re
from datasets import load_from_disk

dataset = load_from_disk("./train_dataset/")


def remove_characters(text, char_types):
    selected_ranges = [ranges[t] for t in char_types if t in ranges]
    if not selected_ranges:
        return text, 0
    pattern = f"[{''.join(selected_ranges)}]"
    removed_chars = len(re.findall(pattern, text))
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text, removed_chars


def process_example(example, fields_to_process, char_types):
    removed_chars = 0
    for field in fields_to_process:
        if field in example:
            if isinstance(example[field], str):
                example[field], chars_removed = remove_characters(
                    example[field], char_types
                )
                if field == "context":
                    removed_chars = chars_removed
            elif isinstance(example[field], dict):
                for subkey, value in example[field].items():
                    if isinstance(value, str):
                        example[field][subkey], _ = remove_characters(value, char_types)

    if "answers" in example and "answer_start" in example["answers"]:
        example["answers"]["answer_start"] = [
            max(0, start - removed_chars)
            for start in example["answers"]["answer_start"]
        ]

    return example


def process_dataset(dataset, fields_to_process, char_types):
    """
    Process the dataset by removing specified character types from specified fields.

    Args:
    dataset (Dataset): The dataset to process.
    fields_to_process (list): List of field names to process.
    char_types (list): List of character types to remove.

    Returns:
    Dataset: The processed dataset.
    """
    return dataset.map(
        lambda x: process_example(
            x,
            fields_to_process=fields_to_process,
            char_types=char_types,
        )
    )


# Usage example:
# processed_dataset = process_dataset(dataset, ["title", "context", "question", "answers"], ["cjk_unified_ideographs"])
