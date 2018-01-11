import logging
import multiprocessing as mp
import os

log = logging.getLogger('main')

class LargeFileMultiProcessor(object):
    def __init__(self, file_path, num_process=None, verbose=True):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.verbose = verbose

        # use all cpu cores if not specifed
        if num_process is None:
            self.num_process = mp.cpu_count()
        self.chunk_size_min = int(self.file_size/self.num_process)

    def process(self):
        if self.verbose:
            log.info('')
            log.info('Large file multiprocessor launched.')
            log.info('File : %s' % self.file_path)
            log.info('Size of file : %s' % self.file_size)
            log.info('Number of processes : %s' % self.num_process)

        chunks = []
        with open(self.file_path, "rb") as f:
            for i in range(self.num_process):
                start = f.tell()
                f.seek(self.chunk_size_min, os.SEEK_CUR)
                if f.readline(): # go to the end of the line
                    end = f.tell()
                else:
                    end = f.seek(0, os.SEEK_END)
                chunks.append([i, start, end])

        if self.verbose:
            log.info('Preparing for multiprocessing...')
        pool = mp.Pool(processes=self.num_process)
        pool_results = pool.map(self._process_chunk, chunks)

        pool.close() # no more tasks
        pool.join() # wrap up current tasks
        return pool_results

    def _process_chunk(self, chunks):
        raise NotImplementedError


class LineCounter(LargeFileMultiProcessor):
    @classmethod
    def count(cls, file_path, num_process=None):
        processor = cls(file_path, num_process, verbose=False)
        pool_results =  processor.process()
        return sum(pool_results)

    def _blocks(self, f, start, end, read_size=64*1024): # 65536
        chunk_size = end - start
        _break = False
        while not _break:
            if _break: break
            if (f.tell() + read_size) > (start + chunk_size):
                read_size = int(start + chunk_size - f.tell())
                _break = True
            yield f.read(read_size)

    def _process_chunk(self, chunk):
        i, start, end = chunk
        num_line = 0
        with open(self.file_path, "r") as f:
            f.seek(start)
            for i, block  in enumerate(self._blocks(f, start, end)):
                num_line +=  block.count('\n')
        return num_line
