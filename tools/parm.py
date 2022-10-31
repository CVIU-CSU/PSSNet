parm = {}
for name, parameters in model.named_parameters():
    parm[name] = parameters.detach().numpy()
with open('parm.txt', 'w') as f:
    f.write(str(parm['generator.decode_head.0.conv_seg.weight'].reshape(3, 512)) + '\n')
    f.write(str(parm['generator.decode_head.0.conv_seg.bias']) + '\n')
    f.write(str(parm['generator.decode_head.0.conv_seg_multi.weight'].reshape(4, 512)) + '\n')
    f.write(str(parm['generator.decode_head.0.conv_seg_multi.bias']) + '\n')
    f.write(str(parm['generator.decode_head.1.conv_seg.weight'].reshape(4, 512)) + '\n')
    f.write(str(parm['generator.decode_head.1.conv_seg.bias']) + '\n')
    f.write(str(parm['generator.decode_head.1.conv_seg_multi.weight'].reshape(3, 512)) + '\n')
    f.write(str(parm['generator.decode_head.1.conv_seg_multi.bias']) + '\n')
