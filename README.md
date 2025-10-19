# SETUP THE PROJECT


### 1 - Config the database

Update the .env


### 2 - Add the files on knowledge and run the fonction 
    MAIN_PIPELINE_EMBEDDING_process_pdfs_to_pgvector()


### 3 - 




##### LAUNCH POSTGRES USING DOCKER

docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16


