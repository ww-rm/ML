import scipy.sparse
import numpy

a = scipy.sparse.csr_matrix(
    [[1, 2, 3]]
)

b = numpy.array(
    [1, 2, 3]
)

print(a.dot(b))
