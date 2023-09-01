from onnx_vectorize import vectorize

file_path = '/depot/davisjam/data/chingwo/PTM-Naming/models/resnet18-v1-7.onnx'

d, l, p = vectorize(file_path)

print('Dimension Vector')
print(d)
print('Layer Vector')
print(l)
print('Parameter Vector')
print(p)