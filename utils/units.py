def pretty_bytes(num_bytes):
    """Convert a byte count into a human-readable string with appropriate units."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1000.0:
            if num_bytes % 1 == 0:
                return f"{int(num_bytes)} {unit}"
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1000.0
    return f"{num_bytes:.2f} PB"

def pretty_bits(num_bits):
    """Convert a bit count into a human-readable string with appropriate units."""
    for unit in ['b', 'Kb', 'Mb', 'Gb', 'Tb']:
        if num_bits < 1000.0:
            if num_bits % 1 == 0:
                return f"{int(num_bits)} {unit}"
            return f"{num_bits:.2f} {unit}"
        num_bits /= 1000.0
    return f"{num_bits:.2f} Pb"

def pretty_SI(num):
    """Convert a large number into a human-readable string with SI prefixes."""
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if abs(num) < 1000.0:
            if num % 1 == 0:
                return f"{int(num)} {unit}"
            return f"{num:.2f} {unit}"
        num /= 1000.0
    return f"{num:.2f} E"