# save_path=$HOME/data/searchR1

index_file=/data/home/zdhs0006/data/retrieval_data/e5_Flat.index
corpus_file=/data/home/zdhs0006/data/retrieval_data/wiki-18.jsonl
retriever_name=e5
retriever_path=/data/home/zdhs0006/data/retrieval_data/e5-base-v2

python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu \
  --port 8005 \