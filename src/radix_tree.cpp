#include "kvlearn/radix_tree.hpp"

namespace kvlearn {

std::vector<std::string> RadixTree::split_key(const std::string& key) {
  // Split on '/' for hierarchical prefixes: "sys/img42/q7"
  std::vector<std::string> parts;
  std::string cur;
  for (char c : key) {
    if (c == '/') {
      if (!cur.empty()) {
        parts.push_back(cur);
        cur.clear();
      }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) parts.push_back(cur);
  if (parts.empty()) parts.push_back(key);
  return parts;
}

RadixTree::PathInfo RadixTree::upsert(const std::string& prefix_key,
                                      BlockId id) {
  PathInfo info;
  Node* cur = root_.get();
  const auto parts = split_key(prefix_key);
  for (const auto& p : parts) {
    auto& slot = cur->children[p];
    if (!slot) {
      slot = std::make_unique<Node>();
      slot->edge = p;
      slot->depth = cur->depth + 1;
    }
    cur = slot.get();
  }
  cur->block_id = id;
  ++cur->access_count;
  info.depth = cur->depth;
  info.num_children = static_cast<int>(cur->children.size());
  info.node = cur;
  return info;
}

RadixTree::PathInfo RadixTree::lookup(const std::string& prefix_key) const {
  PathInfo info;
  const Node* cur = root_.get();
  const auto parts = split_key(prefix_key);
  for (const auto& p : parts) {
    auto it = cur->children.find(p);
    if (it == cur->children.end()) {
      info.node = nullptr;
      return info;
    }
    cur = it->second.get();
  }
  info.depth = cur->depth;
  info.num_children = static_cast<int>(cur->children.size());
  info.node = const_cast<Node*>(cur);
  return info;
}

void RadixTree::clear_block(const std::string& prefix_key) {
  auto info = lookup(prefix_key);
  if (info.node) info.node->block_id = 0;
}

}  // namespace kvlearn
