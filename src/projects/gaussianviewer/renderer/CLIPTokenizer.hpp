#pragma once

#include <array>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <locale>
#include <codecvt>
#include <cctype>

class ReplicatedTokenizer {
private:
    int context_length;
    std::array<wchar_t, 256> byte_encoder;
    std::map<wchar_t, unsigned char> byte_decoder;
    std::map<std::wstring, int> encoder;
    std::map<int, std::wstring> decoder;
    std::map<std::pair<std::wstring, std::wstring>, int> bpe_ranks;
    std::map<std::wstring, std::wstring> cache;
    std::wregex pat;
    int sot_token_id;
    int eot_token_id;

    // Private helper methods
    std::array<wchar_t, 256> bytes_to_unicode();
    std::set<std::pair<std::wstring, std::wstring>> get_pairs(const std::vector<std::wstring>& word);
    std::wstring basic_clean(const std::wstring& text);
    std::wstring whitespace_clean(const std::wstring& text);
    std::wstring bpe(const std::wstring& token);
    std::vector<int> encode(const std::wstring& text);

public:
    // Constructor
    ReplicatedTokenizer(const std::string& merges_file_path, int ctx_len = 77);

    // Main tokenization operator
    std::vector<std::vector<int>> operator()(const std::vector<std::wstring>& texts, int ctx_len = -1);

    // Getter for eot_token_id
    int get_eot_token_id() const;
};