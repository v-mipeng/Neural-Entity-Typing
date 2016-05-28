

def eng_filter(samples):
    '''
    Filter out samples contain non-english characters.
    '''
    samples_copy = []
    for sample in samples:
        try:
            sample[1].decode('utf-8').encode('ascii')
            samples_copy.append(sample)
        except:
            pass
    return samples_copy

