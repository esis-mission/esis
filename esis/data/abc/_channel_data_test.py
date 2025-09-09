import IPython.display
from msfc_ccd._images._tests.test_images import AbstractTestAbstractImageData
import esis


class AbstractTestAbstractChannelData(
    AbstractTestAbstractImageData,
):

    def test_to_jshtml(self, a: esis.data.abc.AbstractChannelData):
        index = {
            a.axis_time: slice(0, 1),
            a.axis_x: slice(0, 100),
            a.axis_y: slice(0, 100),
        }
        a = a[index]
        result = a.to_jshtml()
        assert isinstance(result, IPython.display.HTML)
