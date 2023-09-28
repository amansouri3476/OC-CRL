for file in $(find . -name 'offline*'); do
 wandb sync  $file
 sleep 2
done
