#!/usr/bin/env bash

wiki_dump_file=$1
vocabulary_file=$2
resource_prefix=$3
path_support=$4

echo '******************'
echo $wiki_dump_file
echo $vocabulary_file
echo $resource_prefix
echo $path_support
echo '******************'
# Parse wikipedia. Splitting to 20 files and running in parallel.
echo 'Parsing wikipedia...'
split -nl/10 $wiki_dump_file $wiki_dump_file"_";

for x in {a..j}
do
 ( python parse_wikipedia.py $wiki_dump_file"_a"$x $vocabulary_file $wiki_dump_file"_a"$x"_parsed" ) &
done
wait

# triplet_file="wiki_parsed"
# cat $wiki_dump_file"_a"*"_parsed" > $triplet_file
# echo 'omit parsing...'
# Create the frequent paths file (take paths that occurred approximately at least 5 times. To take paths that occurred with at least 5 different pairs,
# replace with the commented lines - consumes much more memory).
#sort -u $triplet_file | cut -f3 -d$'\t' > paths;
#awk -F$'\t' '{a[$1]++; if (a[$1] == 5) print $1}' paths > frequent_paths;
#rm paths;
for x in {a..j}
do
( awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' $wiki_dump_file"_a"$x"_parsed" > paths"_a"$x ) &
done
wait

cat paths_a* > paths_temp;
cat paths_temp | grep -v "$(printf '\t1$')" > frequent_paths_temp;
awk -F$'\t' '{i[$1]+=$2} END{for(x in i){print x"\t"i[x]}}' frequent_paths_temp > paths;
awk -v var="$path_support" -F$'\t' '$2 >= var {print $1}' paths > frequent_paths;
rm paths_temp frequent_paths_temp paths_a*; # You can remove paths to save space, or keep it to change the threshold for frequent paths

# First step - create the term and path to ID dictionaries
echo 'Creating the resource from the triplets file...'
python create_resource_from_corpus_1.py frequent_paths $vocabulary_file $resource_prefix;

# Second step - convert the textual triplets to triplets of IDs. 
for x in {a..j}
do
( python create_resource_from_corpus_2.py $wiki_dump_file"_a"$x"_parsed" $resource_prefix ) &
done
wait

# Third step - use the ID-based triplet file and converts it to the '_l2r.db' file
for x in {a..j}
do
( awk -v OFS='\t' '{i[$0]++} END{for(x in i){print x, i[x]}}' $wiki_dump_file"_a"$x"_parsed_id" > id_triplet_file"_a"$x ) &
done
wait

cat id_triplet_file_a* > id_triplet_file_temp;
rm id_triplet_file_a*;

awk -F $'\t' 'BEGIN { OFS = FS } { if($1 % 5 == 0) {a[$1][$2][$3]+=$4; } } END {for (i in a) for (j in a[i]) for (k in a[i][j]) print i, j, k, a[i][j][k]}' id_triplet_file_temp > id_triplet_file_0 &
awk -F $'\t' 'BEGIN { OFS = FS } { if($1 % 5 == 1) {a[$1][$2][$3]+=$4; } } END {for (i in a) for (j in a[i]) for (k in a[i][j]) print i, j, k, a[i][j][k]}' id_triplet_file_temp > id_triplet_file_1 &
awk -F $'\t' 'BEGIN { OFS = FS } { if($1 % 5 == 2) {a[$1][$2][$3]+=$4; } } END {for (i in a) for (j in a[i]) for (k in a[i][j]) print i, j, k, a[i][j][k]}' id_triplet_file_temp > id_triplet_file_2 &
awk -F $'\t' 'BEGIN { OFS = FS } { if($1 % 5 == 3) {a[$1][$2][$3]+=$4; } } END {for (i in a) for (j in a[i]) for (k in a[i][j]) print i, j, k, a[i][j][k]}' id_triplet_file_temp > id_triplet_file_3 &
awk -F $'\t' 'BEGIN { OFS = FS } { if($1 % 5 == 4) {a[$1][$2][$3]+=$4; } } END {for (i in a) for (j in a[i]) for (k in a[i][j]) print i, j, k, a[i][j][k]}' id_triplet_file_temp > id_triplet_file_4 &
wait

cat id_triplet_file_* > id_triplet_file;

python create_resource_from_corpus_3.py id_triplet_file $resource_prefix;

# You can delete triplet_file now and keep only id_triplet_file which is more efficient, or delete both.
# rm id_triplet_file_*
# rm $wiki_dump_file"_a"*

mkdir $resource_prefix
mv $resource_prefix*.db $resource_prefix
mv -t $resource_prefix paths id_triplet_file frequent_paths wiki_parsed 
