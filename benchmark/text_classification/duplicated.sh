cd output_activations/lora

rm *.self_input.pt
rm *.output_input.pt
rm *.attention_input.pt
rm *.intermediate_input.pt
rm classifier_input.pt

for id in 0 1 2 3 4 5 6 7 8 9 10 11
do
  rm roberta.encoder.layer.${id}_input.pt
done