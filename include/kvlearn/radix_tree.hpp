#pragma once

#include "kvlearn/types.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace kvlearn {

// Lightweight radix / prefix metadata tree for feature extraction.
// Stores only control-plane counters; not the KV tensors themselves.
class RadixTree {
 public:
  struct Node {
    std::string edge;           // token-chunk key segment
    BlockId block_id = 0;       // 0 = no resident block at this node
    int depth = 0;
    int access_count = 0;
    std::unordered_map<std::string, std::unique_ptr<Node>> children;
  };

  RadixTree() : root_(std::make_unique<Node>()) { root_->depth = 0; }

  // Insert / update path for prefix_key; returns (depth, num_children).
  struct PathInfo {
    int depth = 0;
    int num_children = 0;
    Node* node = nullptr;
  };
  PathInfo upsert(const std::string& prefix_key, BlockId id);
  PathInfo lookup(const std::string& prefix_key) const;
  void clear_block(const std::string& prefix_key);

  Node* root() { return root_.get(); }
  const Node* root() const { return root_.get(); }

 private:
  std::unique_ptr<Node> root_;

  static std::vector<std::string> split_key(const std::string& key);
};

}  // namespace kvlearn
