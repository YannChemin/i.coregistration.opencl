/****************************************************************************
 *
 * MODULE:       i.coregistration.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Minimal reader for PyTorch checkpoints, in the torch.save
 *               zip format and in the legacy format of torch < 1.6
 *               (DISK's published weights): parses the state_dict pickle
 *               and locates each storage, without any Python dependency.
 *               Also reads safetensors files (Hugging Face weights): a
 *               JSON header followed by contiguous tensors.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "progress.h"
#include "pth_loader.h"

/* Zip archive handling. torch.save() writes stored (uncompressed)
 * entries, so an entry's payload can be read in place once the local
 * header has been skipped. */

struct zip_entry {
    char *name;
    int method;
    uint64_t size;
    uint64_t local_offset;
};

struct zip_dir {
    int n;
    struct zip_entry *e;
};

static uint16_t rd16(const unsigned char *p)
{
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const unsigned char *p)
{
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

static void read_at(FILE *fp, off_t off, void *buf, size_t n, const char *path)
{
    if (fseeko(fp, off, SEEK_SET) != 0 || fread(buf, 1, n, fp) != n)
        G_fatal_error(_("Unable to read %lu bytes at offset %lld in <%s>"),
                      (unsigned long)n, (long long)off, path);
    progress_bytes(n);
}

static void zip_read_dir(FILE *fp, const char *path, struct zip_dir *zd)
{
    unsigned char *tail, *cd, *p;
    off_t fsize, tail_off, eocd;
    size_t tail_len;
    uint64_t cd_off, cd_size, nent;
    long i;

    if (fseeko(fp, 0, SEEK_END) != 0)
        G_fatal_error(_("Unable to seek in <%s>"), path);
    fsize = ftello(fp);
    tail_len = fsize < 65557 ? (size_t)fsize : 65557;
    tail_off = fsize - (off_t)tail_len;
    tail = G_malloc(tail_len);
    read_at(fp, tail_off, tail, tail_len, path);

    eocd = -1;
    for (i = (long)tail_len - 22; i >= 0; i--) {
        if (rd32(tail + i) == 0x06054b50) {
            eocd = i;
            break;
        }
    }
    if (eocd < 0)
        G_fatal_error(_("<%s> is not a zip archive (PyTorch >= 1.6 "
                        "checkpoint format expected)"),
                      path);

    nent = rd16(tail + eocd + 10);
    cd_size = rd32(tail + eocd + 12);
    cd_off = rd32(tail + eocd + 16);

    if (nent == 0xFFFF || cd_size == 0xFFFFFFFF || cd_off == 0xFFFFFFFF) {
        /* Zip64: the locator sits right before the classic record. */
        unsigned char loc[20], rec[56];

        if (eocd < 20 || rd32(tail + eocd - 20) != 0x07064b50)
            G_fatal_error(_("Corrupt zip64 directory in <%s>"), path);
        memcpy(loc, tail + eocd - 20, 20);
        read_at(fp, (off_t)rd64(loc + 8), rec, sizeof(rec), path);
        if (rd32(rec) != 0x06064b50)
            G_fatal_error(_("Corrupt zip64 end record in <%s>"), path);
        nent = rd64(rec + 32);
        cd_size = rd64(rec + 40);
        cd_off = rd64(rec + 48);
    }
    G_free(tail);

    cd = G_malloc(cd_size);
    read_at(fp, (off_t)cd_off, cd, cd_size, path);

    zd->n = (int)nent;
    zd->e = G_calloc(nent, sizeof(struct zip_entry));
    p = cd;
    for (i = 0; i < (long)nent; i++) {
        struct zip_entry *e = &zd->e[i];
        unsigned nlen, elen, clen;
        uint32_t csize, usize, loff;
        const unsigned char *x, *xend;

        if (p + 46 > cd + cd_size || rd32(p) != 0x02014b50)
            G_fatal_error(_("Corrupt zip central directory in <%s>"), path);
        e->method = rd16(p + 10);
        csize = rd32(p + 20);
        usize = rd32(p + 24);
        nlen = rd16(p + 28);
        elen = rd16(p + 30);
        clen = rd16(p + 32);
        loff = rd32(p + 42);
        e->name = G_malloc(nlen + 1);
        memcpy(e->name, p + 46, nlen);
        e->name[nlen] = '\0';
        e->size = usize;
        e->local_offset = loff;

        /* Zip64 extra field carries the 64-bit values that overflowed. */
        x = p + 46 + nlen;
        xend = x + elen;
        while (x + 4 <= xend) {
            unsigned id = rd16(x), len = rd16(x + 2);
            const unsigned char *v = x + 4;

            if (id == 0x0001) {
                if (usize == 0xFFFFFFFF) {
                    e->size = rd64(v);
                    v += 8;
                }
                if (csize == 0xFFFFFFFF)
                    v += 8;
                if (loff == 0xFFFFFFFF)
                    e->local_offset = rd64(v);
            }
            x += 4 + len;
        }
        p += 46 + nlen + elen + clen;
    }
    G_free(cd);
}

static off_t zip_data_offset(FILE *fp, const char *path,
                             const struct zip_entry *e)
{
    unsigned char h[30];

    if (e->method != 0)
        G_fatal_error(_("Zip entry <%s> in <%s> is compressed; only stored "
                        "entries (as written by torch.save) are supported"),
                      e->name, path);
    read_at(fp, (off_t)e->local_offset, h, sizeof(h), path);
    if (rd32(h) != 0x04034b50)
        G_fatal_error(_("Corrupt local header for <%s> in <%s>"), e->name,
                      path);
    return (off_t)e->local_offset + 30 + rd16(h + 26) + rd16(h + 28);
}

/* Pickle virtual machine, restricted to what torch.save() emits for a
 * state_dict: dicts/OrderedDicts of tensors rebuilt from persistent
 * storage references. Unknown callables yield opaque objects; unknown
 * opcodes are fatal. */

enum pv_type {
    PV_NONE,
    PV_BOOL,
    PV_INT,
    PV_FLOAT,
    PV_STR,
    PV_TUPLE,
    PV_LIST,
    PV_DICT,
    PV_GLOBAL,
    PV_STORAGE,
    PV_TENSOR,
    PV_MARK,
    PV_OBJ
};

struct pv {
    enum pv_type t;
    long i;
    double f;
    char *s;  /* PV_STR text, PV_GLOBAL module, PV_STORAGE key. */
    char *s2; /* PV_GLOBAL name. */
    int n, cap;
    struct pv **items; /* Tuple/list items; dict as k0, v0, k1, v1... */
    enum pth_dtype dtype;
    int elsize; /* PV_STORAGE element size in bytes. */
    struct pv *storage;
    int ndim;
    long shape[PTH_MAX_DIMS], stride[PTH_MAX_DIMS];
};

struct pvm {
    const unsigned char *p, *end;
    struct pv **stack;
    int sp, scap;
    struct pv **memo;
    long mcap;
    struct pv **all; /* Every allocated value, for cleanup. */
    long nall, allcap;
};

static struct pv *pv_new(struct pvm *m, enum pv_type t)
{
    struct pv *v = G_calloc(1, sizeof(struct pv));

    v->t = t;
    if (m->nall == m->allcap) {
        m->allcap = m->allcap ? 2 * m->allcap : 4096;
        m->all = G_realloc(m->all, m->allcap * sizeof(struct pv *));
    }
    m->all[m->nall++] = v;
    return v;
}

static void pv_append(struct pv *c, struct pv *v)
{
    if (c->n == c->cap) {
        c->cap = c->cap ? 2 * c->cap : 8;
        c->items = G_realloc(c->items, c->cap * sizeof(struct pv *));
    }
    c->items[c->n++] = v;
}

static void push(struct pvm *m, struct pv *v)
{
    if (m->sp == m->scap) {
        m->scap = m->scap ? 2 * m->scap : 256;
        m->stack = G_realloc(m->stack, m->scap * sizeof(struct pv *));
    }
    m->stack[m->sp++] = v;
}

static struct pv *pop(struct pvm *m)
{
    if (m->sp <= 0)
        G_fatal_error(_("Pickle stack underflow in checkpoint"));
    return m->stack[--m->sp];
}

static struct pv *top(struct pvm *m)
{
    if (m->sp <= 0)
        G_fatal_error(_("Pickle stack underflow in checkpoint"));
    return m->stack[m->sp - 1];
}

/* Return the stack index of the topmost mark and discard the mark. */
static int pop_mark(struct pvm *m)
{
    int k;

    for (k = m->sp - 1; k >= 0; k--)
        if (m->stack[k]->t == PV_MARK)
            return k;
    G_fatal_error(_("Pickle mark not found in checkpoint"));
    return -1;
}

static const unsigned char *take(struct pvm *m, size_t n)
{
    const unsigned char *q = m->p;

    if ((size_t)(m->end - m->p) < n)
        G_fatal_error(_("Truncated pickle in checkpoint"));
    m->p += n;
    return q;
}

static char *take_line(struct pvm *m)
{
    const unsigned char *q = m->p;
    char *s;

    while (m->p < m->end && *m->p != '\n')
        m->p++;
    if (m->p >= m->end)
        G_fatal_error(_("Truncated pickle in checkpoint"));
    s = G_malloc(m->p - q + 1);
    memcpy(s, q, m->p - q);
    s[m->p - q] = '\0';
    m->p++;
    return s;
}

static void memo_put(struct pvm *m, long idx, struct pv *v)
{
    if (idx < 0)
        G_fatal_error(_("Invalid pickle memo index"));
    if (idx >= m->mcap) {
        long ncap = m->mcap ? m->mcap : 1024;

        while (ncap <= idx)
            ncap *= 2;
        m->memo = G_realloc(m->memo, ncap * sizeof(struct pv *));
        memset(m->memo + m->mcap, 0, (ncap - m->mcap) * sizeof(struct pv *));
        m->mcap = ncap;
    }
    m->memo[idx] = v;
}

static struct pv *memo_get(struct pvm *m, long idx)
{
    if (idx < 0 || idx >= m->mcap || !m->memo[idx])
        G_fatal_error(_("Invalid pickle memo reference %ld"), idx);
    return m->memo[idx];
}

static struct pv *make_str(struct pvm *m, const unsigned char *s, size_t n)
{
    struct pv *v = pv_new(m, PV_STR);

    v->s = G_malloc(n + 1);
    memcpy(v->s, s, n);
    v->s[n] = '\0';
    return v;
}

static struct pv *make_tuple_from(struct pvm *m, int from)
{
    struct pv *v = pv_new(m, PV_TUPLE);
    int k;

    for (k = from; k < m->sp; k++)
        pv_append(v, m->stack[k]);
    m->sp = from;
    return v;
}

static long pv_int(const struct pv *v)
{
    if (v->t == PV_INT || v->t == PV_BOOL)
        return v->i;
    G_fatal_error(_("Integer expected in checkpoint pickle"));
    return 0;
}

static int is_global(const struct pv *v, const char *mod, const char *name)
{
    return v->t == PV_GLOBAL && strcmp(v->s, mod) == 0 &&
           strcmp(v->s2, name) == 0;
}

static struct pv *reduce(struct pvm *m, struct pv *fn, struct pv *args)
{
    struct pv *v;
    int d;

    if (args->t != PV_TUPLE)
        G_fatal_error(_("Pickle REDUCE without argument tuple"));

    if (is_global(fn, "torch._utils", "_rebuild_tensor_v2") ||
        is_global(fn, "torch._utils", "_rebuild_tensor")) {
        struct pv *size, *stride;

        if (args->n < 4 || args->items[0]->t != PV_STORAGE ||
            args->items[2]->t != PV_TUPLE || args->items[3]->t != PV_TUPLE)
            G_fatal_error(_("Unexpected tensor rebuild arguments"));
        size = args->items[2];
        stride = args->items[3];
        if (size->n > PTH_MAX_DIMS || size->n != stride->n)
            G_fatal_error(_("Unsupported tensor rank %d"), size->n);
        v = pv_new(m, PV_TENSOR);
        v->storage = args->items[0];
        v->i = pv_int(args->items[1]);
        v->ndim = size->n;
        for (d = 0; d < size->n; d++) {
            v->shape[d] = pv_int(size->items[d]);
            v->stride[d] = pv_int(stride->items[d]);
        }
        return v;
    }
    if (is_global(fn, "torch._utils", "_rebuild_parameter") ||
        is_global(fn, "torch._utils", "_rebuild_parameter_with_state")) {
        if (args->n < 1 || args->items[0]->t != PV_TENSOR)
            G_fatal_error(_("Unexpected parameter rebuild arguments"));
        return args->items[0];
    }
    /* A tensor subclass (torch.Tensor itself in DINOv2's checkpoint):
     * (rebuild function, type, its arguments, state); the type and state
     * do not change the data. */
    if (is_global(fn, "torch._tensor", "_rebuild_from_type_v2")) {
        if (args->n < 3 || args->items[0]->t != PV_GLOBAL ||
            args->items[2]->t != PV_TUPLE)
            G_fatal_error(_("Unexpected tensor subclass rebuild arguments"));
        return reduce(m, args->items[0], args->items[2]);
    }
    if (is_global(fn, "collections", "OrderedDict"))
        return pv_new(m, PV_DICT);

    return pv_new(m, PV_OBJ);
}

static struct pv *persistent_load(struct pvm *m, struct pv *pid)
{
    struct pv *v, *type;
    const char *tn;

    if (pid->t != PV_TUPLE || pid->n < 5 || pid->items[0]->t != PV_STR ||
        strcmp(pid->items[0]->s, "storage") != 0 ||
        pid->items[1]->t != PV_GLOBAL || pid->items[2]->t != PV_STR)
        G_fatal_error(_("Unsupported persistent id in checkpoint"));

    type = pid->items[1];
    tn = type->s2;
    v = pv_new(m, PV_STORAGE);
    v->elsize = 2;
    if (strcmp(tn, "FloatStorage") == 0) {
        v->dtype = PTH_F32;
        v->elsize = 4;
    }
    else if (strcmp(tn, "HalfStorage") == 0)
        v->dtype = PTH_F16;
    else if (strcmp(tn, "BFloat16Storage") == 0)
        v->dtype = PTH_BF16;
    else {
        /* Other types are indexed (the legacy format needs their size)
         * but their tensors are left out of the state_dict. */
        static const char *names[] = {
            "DoubleStorage", "LongStorage", "IntStorage", "ShortStorage",
            "CharStorage",   "ByteStorage", "BoolStorage"};
        static const int sizes[] = {8, 8, 4, 2, 1, 1, 1};
        int k;

        v->dtype = PTH_OTHER;
        v->elsize = 0;
        for (k = 0; k < (int)(sizeof(sizes) / sizeof(sizes[0])); k++)
            if (strcmp(tn, names[k]) == 0)
                v->elsize = sizes[k];
        if (!v->elsize)
            G_fatal_error(_("Unsupported tensor storage type <%s.%s> in "
                            "checkpoint"),
                          type->s, tn);
    }
    v->s = pid->items[2]->s;
    return v;
}

static struct pv *run_pickle(struct pvm *m)
{
    for (;;) {
        unsigned char op = *take(m, 1);
        struct pv *v, *a, *b;
        int k;

        switch (op) {
        case 0x80: /* PROTO */
            take(m, 1);
            break;
        case 0x95: /* FRAME */
            take(m, 8);
            break;
        case '.': /* STOP */
            return pop(m);
        case '(': /* MARK */
            push(m, pv_new(m, PV_MARK));
            break;
        case '}': /* EMPTY_DICT */
            push(m, pv_new(m, PV_DICT));
            break;
        case ']': /* EMPTY_LIST */
            push(m, pv_new(m, PV_LIST));
            break;
        case ')': /* EMPTY_TUPLE */
            push(m, pv_new(m, PV_TUPLE));
            break;
        case 'N': /* NONE */
            push(m, pv_new(m, PV_NONE));
            break;
        case 0x88: /* NEWTRUE */
        case 0x89: /* NEWFALSE */
            v = pv_new(m, PV_BOOL);
            v->i = op == 0x88;
            push(m, v);
            break;
        case 'J': /* BININT */
            v = pv_new(m, PV_INT);
            v->i = (int32_t)rd32(take(m, 4));
            push(m, v);
            break;
        case 'K': /* BININT1 */
            v = pv_new(m, PV_INT);
            v->i = *take(m, 1);
            push(m, v);
            break;
        case 'M': /* BININT2 */
            v = pv_new(m, PV_INT);
            v->i = rd16(take(m, 2));
            push(m, v);
            break;
        case 0x8a: { /* LONG1 */
            int n = *take(m, 1);
            const unsigned char *q = take(m, n);
            uint64_t u = 0;

            /* Wider integers (the 80-bit magic number of the legacy
             * format) are kept as opaque values. */
            if (n > 8) {
                push(m, pv_new(m, PV_OBJ));
                break;
            }
            for (k = n - 1; k >= 0; k--)
                u = (u << 8) | q[k];
            if (n > 0 && n < 8 && (q[n - 1] & 0x80))
                u |= ~(uint64_t)0 << (8 * n);
            v = pv_new(m, PV_INT);
            v->i = (long)u;
            push(m, v);
            break;
        }
        case 'G': { /* BINFLOAT, big endian */
            const unsigned char *q = take(m, 8);
            uint64_t u = 0;
            double d;

            for (k = 0; k < 8; k++)
                u = (u << 8) | q[k];
            memcpy(&d, &u, 8);
            v = pv_new(m, PV_FLOAT);
            v->f = d;
            push(m, v);
            break;
        }
        case 'X': { /* BINUNICODE */
            uint32_t n = rd32(take(m, 4));

            push(m, make_str(m, take(m, n), n));
            break;
        }
        case 0x8c: { /* SHORT_BINUNICODE */
            unsigned n = *take(m, 1);

            push(m, make_str(m, take(m, n), n));
            break;
        }
        case 0x8d: { /* BINUNICODE8 */
            uint64_t n = rd64(take(m, 8));

            push(m, make_str(m, take(m, n), n));
            break;
        }
        case 'T': { /* BINSTRING */
            uint32_t n = rd32(take(m, 4));

            push(m, make_str(m, take(m, n), n));
            break;
        }
        case 'U': { /* SHORT_BINSTRING */
            unsigned n = *take(m, 1);

            push(m, make_str(m, take(m, n), n));
            break;
        }
        case 'c': /* GLOBAL */
            v = pv_new(m, PV_GLOBAL);
            v->s = take_line(m);
            v->s2 = take_line(m);
            push(m, v);
            break;
        case 0x93: /* STACK_GLOBAL */
            b = pop(m);
            a = pop(m);
            if (a->t != PV_STR || b->t != PV_STR)
                G_fatal_error(_("Invalid STACK_GLOBAL in checkpoint"));
            v = pv_new(m, PV_GLOBAL);
            v->s = G_store(a->s);
            v->s2 = G_store(b->s);
            push(m, v);
            break;
        case 'q': /* BINPUT */
            memo_put(m, *take(m, 1), top(m));
            break;
        case 'r': /* LONG_BINPUT */
            memo_put(m, rd32(take(m, 4)), top(m));
            break;
        case 0x94: /* MEMOIZE */
        {
            long idx = 0;

            while (idx < m->mcap && m->memo[idx])
                idx++;
            memo_put(m, idx, top(m));
            break;
        }
        case 'h': /* BINGET */
            push(m, memo_get(m, *take(m, 1)));
            break;
        case 'j': /* LONG_BINGET */
            push(m, memo_get(m, rd32(take(m, 4))));
            break;
        case 't': /* TUPLE */
            k = pop_mark(m);
            v = make_tuple_from(m, k + 1);
            m->sp = k;
            push(m, v);
            break;
        case 0x85: /* TUPLE1 */
        case 0x86: /* TUPLE2 */
        case 0x87: /* TUPLE3 */
            k = m->sp - (op - 0x84);
            if (k < 0)
                G_fatal_error(_("Pickle stack underflow in checkpoint"));
            push(m, make_tuple_from(m, k));
            break;
        case 'Q': /* BINPERSID */
            push(m, persistent_load(m, pop(m)));
            break;
        case 'R': /* REDUCE */
            a = pop(m);
            b = pop(m);
            push(m, reduce(m, b, a));
            break;
        case 0x81: /* NEWOBJ */
            pop(m);
            pop(m);
            push(m, pv_new(m, PV_OBJ));
            break;
        case 'b': /* BUILD: state is irrelevant for a state_dict. */
            pop(m);
            break;
        case 's': /* SETITEM */
            b = pop(m);
            a = pop(m);
            v = top(m);
            if (v->t == PV_DICT) {
                pv_append(v, a);
                pv_append(v, b);
            }
            break;
        case 'u': /* SETITEMS */
            k = pop_mark(m);
            if (k < 1 || ((m->sp - k - 1) % 2) != 0)
                G_fatal_error(_("Invalid SETITEMS in checkpoint pickle"));
            v = m->stack[k - 1];
            if (v->t == PV_DICT) {
                int j;

                for (j = k + 1; j < m->sp; j++)
                    pv_append(v, m->stack[j]);
            }
            m->sp = k;
            break;
        case 'a': /* APPEND */
            a = pop(m);
            v = top(m);
            if (v->t == PV_LIST)
                pv_append(v, a);
            break;
        case 'e': /* APPENDS */
            k = pop_mark(m);
            if (k < 1)
                G_fatal_error(_("Invalid APPENDS in checkpoint pickle"));
            v = m->stack[k - 1];
            if (v->t == PV_LIST) {
                int j;

                for (j = k + 1; j < m->sp; j++)
                    pv_append(v, m->stack[j]);
            }
            m->sp = k;
            break;
        default:
            G_fatal_error(_("Unsupported pickle opcode 0x%02x in checkpoint"),
                          op);
        }
    }
}

static int dict_has_tensors(const struct pv *d)
{
    int k;

    for (k = 1; k < d->n; k += 2)
        if (d->items[k]->t == PV_TENSOR)
            return 1;
    return 0;
}

/* Accept a bare state_dict, or a wrapper dict holding one under "model"
 * or "state_dict" (common for training checkpoints), or under the only
 * key whose value is a dict of tensors ("extractor" for DISK). */
static const struct pv *find_state_dict(const struct pv *root)
{
    int k;

    if (root->t != PV_DICT)
        return NULL;
    if (dict_has_tensors(root))
        return root;
    for (k = 0; k + 1 < root->n; k += 2) {
        const struct pv *key = root->items[k], *val = root->items[k + 1];

        if (key->t == PV_STR && val->t == PV_DICT &&
            (strcmp(key->s, "model") == 0 ||
             strcmp(key->s, "state_dict") == 0) &&
            dict_has_tensors(val))
            return val;
    }
    {
        const struct pv *only = NULL;
        int n = 0;

        for (k = 0; k + 1 < root->n; k += 2) {
            const struct pv *key = root->items[k], *val = root->items[k + 1];

            if (key->t == PV_STR && val->t == PV_DICT &&
                dict_has_tensors(val)) {
                only = val;
                n++;
            }
        }
        if (n == 1)
            return only;
    }
    return NULL;
}

static size_t dtype_size(enum pth_dtype d) { return d == PTH_F32 ? 4 : 2; }

/* Where each storage of the checkpoint lives in the file. */
struct storage_index {
    int n;
    char **key;
    off_t *offset;
    size_t *bytes;
};

static void index_add(struct storage_index *ix, const char *key, off_t off,
                      size_t bytes)
{
    ix->key = G_realloc(ix->key, (ix->n + 1) * sizeof(char *));
    ix->offset = G_realloc(ix->offset, (ix->n + 1) * sizeof(off_t));
    ix->bytes = G_realloc(ix->bytes, (ix->n + 1) * sizeof(size_t));
    ix->key[ix->n] = G_store(key);
    ix->offset[ix->n] = off;
    ix->bytes[ix->n++] = bytes;
}

static int index_find(const struct storage_index *ix, const char *key)
{
    int i;

    for (i = 0; i < ix->n; i++)
        if (strcmp(ix->key[i], key) == 0)
            return i;
    return -1;
}

static void index_free(struct storage_index *ix)
{
    int i;

    for (i = 0; i < ix->n; i++)
        G_free(ix->key[i]);
    G_free(ix->key);
    G_free(ix->offset);
    G_free(ix->bytes);
}

static void pvm_free(struct pvm *m)
{
    int i;

    for (i = 0; i < m->nall; i++) {
        struct pv *v = m->all[i];

        /* PV_STORAGE keys alias PV_STR payloads and are not freed. */
        if (v->t == PV_STR || v->t == PV_GLOBAL)
            G_free(v->s);
        if (v->t == PV_GLOBAL)
            G_free(v->s2);
        G_free(v->items);
    }
    for (i = 0; i < m->nall; i++)
        G_free(m->all[i]);
    G_free(m->all);
    G_free(m->stack);
    G_free(m->memo);
}

/* Start a new pickle on the same machine: an empty stack and memo (memo
 * indices restart in every pickle), values kept for cleanup. */
static void pvm_next(struct pvm *m)
{
    m->sp = 0;
    if (m->memo)
        memset(m->memo, 0, m->mcap * sizeof(struct pv *));
}

/* torch.save zip format: data.pkl holds the object, data/<key> each
 * storage. Returns the state_dict; m keeps its values alive. */
static const struct pv *open_zip(struct pth_file *pf, const char *path,
                                 struct pvm *m, unsigned char **buf,
                                 struct storage_index *ix)
{
    struct zip_dir zd;
    const struct zip_entry *pkl = NULL;
    const struct pv *sd;
    char *prefix;
    size_t plen;
    int i;

    zip_read_dir(pf->fp, path, &zd);

    /* The archive root folder is named after the saved file, so locate
     * data.pkl by suffix and derive the prefix from it. */
    for (i = 0; i < zd.n; i++) {
        size_t n = strlen(zd.e[i].name);

        if (n >= 8 && strcmp(zd.e[i].name + n - 8, "data.pkl") == 0 &&
            (n == 8 || zd.e[i].name[n - 9] == '/')) {
            pkl = &zd.e[i];
            break;
        }
    }
    if (!pkl)
        G_fatal_error(_("No data.pkl in <%s>: not a PyTorch checkpoint"), path);
    plen = strlen(pkl->name) - 8;
    prefix = G_malloc(plen + 6);
    memcpy(prefix, pkl->name, plen);
    strcpy(prefix + plen, "data/");

    *buf = G_malloc(pkl->size);
    read_at(pf->fp, zip_data_offset(pf->fp, path, pkl), *buf, pkl->size, path);
    m->p = *buf;
    m->end = *buf + pkl->size;
    sd = find_state_dict(run_pickle(m));

    for (i = 0; i < zd.n; i++)
        if (strncmp(zd.e[i].name, prefix, plen + 5) == 0)
            index_add(ix, zd.e[i].name + plen + 5,
                      zip_data_offset(pf->fp, path, &zd.e[i]), zd.e[i].size);

    G_free(prefix);
    for (i = 0; i < zd.n; i++)
        G_free(zd.e[i].name);
    G_free(zd.e);
    return sd;
}

/* Legacy format (torch < 1.6, or _use_new_zipfile_serialization=False):
 * five pickles (magic number, protocol version, system information, the
 * object, the list of storage keys), then every storage in the order of
 * that list as a little-endian int64 element count and the raw
 * elements. */
static const struct pv *open_legacy(struct pth_file *pf, const char *path,
                                    struct pvm *m, unsigned char **buf,
                                    struct storage_index *ix)
{
    const struct pv *root, *sd, *keys;
    long size;
    off_t off;
    int i, k;

    if (fseeko(pf->fp, 0, SEEK_END) != 0 || (size = ftello(pf->fp)) <= 0)
        G_fatal_error(_("Unable to read checkpoint <%s>"), path);
    *buf = G_malloc(size);
    read_at(pf->fp, 0, *buf, size, path);
    if ((*buf)[0] != 0x80)
        G_fatal_error(_("<%s> is neither a zip archive nor a legacy pickle: "
                        "not a PyTorch checkpoint"),
                      path);
    m->p = *buf;
    m->end = *buf + size;
    run_pickle(m); /* Magic number. */
    pvm_next(m);
    root = run_pickle(m); /* Protocol version. */
    if (root->t != PV_INT || root->i != 1001)
        G_fatal_error(_("<%s> is neither a zip archive nor a legacy PyTorch "
                        "checkpoint"),
                      path);
    pvm_next(m);
    run_pickle(m); /* System information. */
    pvm_next(m);
    sd = find_state_dict(run_pickle(m));
    pvm_next(m);
    keys = run_pickle(m);
    if (keys->t != PV_LIST)
        G_fatal_error(_("<%s>: no storage key list after the object"), path);

    off = m->p - *buf;
    for (i = 0; i < keys->n; i++) {
        const struct pv *key = keys->items[i];
        size_t elsize = 0;
        uint64_t numel;

        if (key->t != PV_STR)
            G_fatal_error(_("<%s>: invalid storage key"), path);
        /* The element size comes from the storage type the object named. */
        for (k = 0; k < m->nall; k++)
            if (m->all[k]->t == PV_STORAGE &&
                strcmp(m->all[k]->s, key->s) == 0) {
                elsize = m->all[k]->elsize;
                break;
            }
        if (!elsize)
            G_fatal_error(_("<%s>: storage <%s> is not used by any tensor"),
                          path, key->s);
        if (off + 8 > size)
            G_fatal_error(_("Truncated checkpoint <%s>"), path);
        numel = rd64(*buf + off);
        off += 8;
        if (off + (off_t)(numel * elsize) > size)
            G_fatal_error(_("Truncated checkpoint <%s>"), path);
        index_add(ix, key->s, off, numel * elsize);
        off += numel * elsize;
    }
    return sd;
}

/* safetensors: an 8-byte little-endian header length, a JSON object
 * mapping tensor names to {"dtype", "shape", "data_offsets"} (plus an
 * optional "__metadata__"), then the data. A small parser for exactly
 * that structure; tensors of other dtypes than F32, F16 and BF16 (such
 * as BatchNorm's I64 counters) are skipped. */
struct json {
    const char *p, *end, *path;
};

static void json_ws(struct json *j)
{
    while (j->p < j->end &&
           (*j->p == ' ' || *j->p == '\n' || *j->p == '\r' || *j->p == '\t'))
        j->p++;
}

static void json_expect(struct json *j, char c)
{
    json_ws(j);
    if (j->p >= j->end || *j->p != c)
        G_fatal_error(_("Malformed safetensors header in <%s>"), j->path);
    j->p++;
}

/* A string without escapes other than \" and \\ (tensor names). */
static char *json_string(struct json *j)
{
    char *out, *o;
    const char *q;

    json_expect(j, '"');
    out = o = G_malloc(j->end - j->p + 1);
    for (q = j->p; q < j->end && *q != '"'; q++) {
        if (*q == '\\' && q + 1 < j->end)
            q++;
        *o++ = *q;
    }
    if (q >= j->end)
        G_fatal_error(_("Malformed safetensors header in <%s>"), j->path);
    *o = '\0';
    j->p = q + 1;
    return out;
}

static long json_int(struct json *j)
{
    char *e;
    long v;

    json_ws(j);
    v = strtol(j->p, &e, 10);
    if (e == j->p)
        G_fatal_error(_("Malformed safetensors header in <%s>"), j->path);
    j->p = e;
    return v;
}

/* Skip any value (the metadata object). */
static void json_skip(struct json *j)
{
    int depth = 0;

    json_ws(j);
    do {
        if (j->p >= j->end)
            G_fatal_error(_("Malformed safetensors header in <%s>"), j->path);
        if (*j->p == '"') {
            G_free(json_string(j));
            continue;
        }
        if (*j->p == '{' || *j->p == '[')
            depth++;
        else if (*j->p == '}' || *j->p == ']')
            depth--;
        j->p++;
        json_ws(j);
    } while (depth > 0 || (j->p < j->end && *j->p != ',' && *j->p != '}'));
}

static int open_safetensors(struct pth_file *pf, const char *path)
{
    unsigned char h8[9];
    uint64_t hlen;
    off_t base;
    char *hdr;
    struct json j;
    int cap = 64;

    if (fseeko(pf->fp, 0, SEEK_END) != 0)
        return 0;
    base = ftello(pf->fp);
    if (base < 9 || fseeko(pf->fp, 0, SEEK_SET) != 0 ||
        fread(h8, 1, 9, pf->fp) != 9)
        return 0;
    hlen = rd64(h8);
    if (h8[8] != '{' || hlen < 2 || (off_t)(8 + hlen) > base)
        return 0;
    hdr = G_malloc(hlen);
    read_at(pf->fp, 8, hdr, hlen, path);
    j.p = hdr;
    j.end = hdr + hlen;
    j.path = path;
    pf->tensors = G_calloc(cap, sizeof(struct pth_tensor));
    json_expect(&j, '{');
    for (;;) {
        char *name, *dtype = NULL;
        long shape[PTH_MAX_DIMS], off0 = -1, off1 = -1;
        int ndim = 0, d;

        json_ws(&j);
        if (j.p < j.end && *j.p == '}')
            break;
        name = json_string(&j);
        json_expect(&j, ':');
        if (strcmp(name, "__metadata__") == 0) {
            json_skip(&j);
            G_free(name);
        }
        else {
            json_expect(&j, '{');
            for (;;) {
                char *key = json_string(&j);

                json_expect(&j, ':');
                if (strcmp(key, "dtype") == 0)
                    dtype = json_string(&j);
                else if (strcmp(key, "shape") == 0 ||
                         strcmp(key, "data_offsets") == 0) {
                    int is_shape = key[0] == 's', n = 0;

                    json_expect(&j, '[');
                    json_ws(&j);
                    while (j.p < j.end && *j.p != ']') {
                        long v = json_int(&j);

                        if (is_shape) {
                            if (n >= PTH_MAX_DIMS)
                                G_fatal_error(_("Tensor <%s> has too many "
                                                "dimensions"),
                                              name);
                            shape[n] = v;
                        }
                        else if (n == 0)
                            off0 = v;
                        else
                            off1 = v;
                        n++;
                        json_ws(&j);
                        if (*j.p == ',')
                            j.p++;
                        json_ws(&j);
                    }
                    json_expect(&j, ']');
                    if (is_shape)
                        ndim = n;
                }
                else
                    json_skip(&j);
                G_free(key);
                json_ws(&j);
                if (*j.p == ',') {
                    j.p++;
                    continue;
                }
                json_expect(&j, '}');
                break;
            }
            if (dtype && off0 >= 0 && off1 >= off0 &&
                (!strcmp(dtype, "F32") || !strcmp(dtype, "F16") ||
                 !strcmp(dtype, "BF16"))) {
                struct pth_tensor *t;
                long n = 1, stride = 1;

                if (pf->ntensors == cap) {
                    cap *= 2;
                    pf->tensors =
                        G_realloc(pf->tensors, cap * sizeof(struct pth_tensor));
                }
                t = &pf->tensors[pf->ntensors++];
                memset(t, 0, sizeof(*t));
                t->name = name;
                t->ndim = ndim;
                t->dtype = !strcmp(dtype, "F32")   ? PTH_F32
                           : !strcmp(dtype, "F16") ? PTH_F16
                                                   : PTH_BF16;
                for (d = 0; d < ndim; d++)
                    n *= shape[d];
                for (d = ndim - 1; d >= 0; d--) {
                    t->shape[d] = shape[d];
                    t->stride[d] = stride;
                    stride *= shape[d];
                }
                t->data_offset = 8 + (off_t)hlen + off0;
                t->storage_bytes = off1 - off0;
                if ((size_t)n * dtype_size(t->dtype) != t->storage_bytes ||
                    t->data_offset + (off_t)t->storage_bytes > base)
                    G_fatal_error(_("Tensor <%s> does not match its data in "
                                    "<%s>"),
                                  name, path);
            }
            else
                G_free(name);
            G_free(dtype);
        }
        json_ws(&j);
        if (j.p < j.end && *j.p == ',')
            j.p++;
    }
    G_free(hdr);
    G_verbose_message(_("safetensors <%s>: %d tensors"), path, pf->ntensors);
    return 1;
}

void pth_open(struct pth_file *pf, const char *path)
{
    const struct pv *sd;
    struct storage_index ix = {0, NULL, NULL, NULL};
    unsigned char sig[4], *buf = NULL;
    struct pvm m;
    int k, nt;

    memset(pf, 0, sizeof(*pf));
    pf->path = G_store(path);
    pf->fp = fopen(path, "rb");
    if (!pf->fp)
        G_fatal_error(_("Unable to open checkpoint <%s>"), path);
    if (fread(sig, 1, 4, pf->fp) != 4)
        G_fatal_error(_("<%s> is too short to be a PyTorch checkpoint"), path);
    if (memcmp(sig, "PK\003\004", 4) != 0 && open_safetensors(pf, path))
        return;
    fseeko(pf->fp, 4, SEEK_SET);

    memset(&m, 0, sizeof(m));
    if (memcmp(sig, "PK\003\004", 4) == 0)
        sd = open_zip(pf, path, &m, &buf, &ix);
    else
        sd = open_legacy(pf, path, &m, &buf, &ix);
    if (!sd)
        G_fatal_error(_("No state_dict of tensors found in <%s>"), path);

    nt = 0;
    for (k = 1; k < sd->n; k += 2)
        if (sd->items[k]->t == PV_TENSOR && sd->items[k - 1]->t == PV_STR &&
            sd->items[k]->storage->dtype != PTH_OTHER)
            nt++;
    pf->tensors = G_calloc(nt, sizeof(struct pth_tensor));

    for (k = 1; k < sd->n; k += 2) {
        const struct pv *key = sd->items[k - 1], *tv = sd->items[k];
        struct pth_tensor *t;
        long last;
        int d, e;

        if (tv->t != PV_TENSOR || key->t != PV_STR ||
            tv->storage->dtype == PTH_OTHER)
            continue;
        t = &pf->tensors[pf->ntensors++];
        t->name = G_store(key->s);
        t->ndim = tv->ndim;
        t->storage_offset = tv->i;
        t->dtype = tv->storage->dtype;
        for (d = 0; d < tv->ndim; d++) {
            t->shape[d] = tv->shape[d];
            t->stride[d] = tv->stride[d];
        }

        e = index_find(&ix, tv->storage->s);
        if (e < 0)
            G_fatal_error(_("Storage <%s> of tensor <%s> missing in <%s>"),
                          tv->storage->s, t->name, path);
        t->data_offset = ix.offset[e];
        t->storage_bytes = ix.bytes[e];

        /* Validate that every strided element lies inside the storage. */
        last = t->storage_offset;
        for (d = 0; d < t->ndim; d++) {
            if (t->shape[d] <= 0)
                G_fatal_error(_("Empty tensor <%s> in checkpoint"), t->name);
            last += (t->shape[d] - 1) * t->stride[d];
        }
        if (t->storage_offset < 0 ||
            (size_t)(last + 1) * dtype_size(t->dtype) > t->storage_bytes)
            G_fatal_error(_("Tensor <%s> exceeds its storage in <%s>"), t->name,
                          path);
    }

    pvm_free(&m);
    index_free(&ix);
    G_free(buf);

    G_verbose_message(_("Checkpoint <%s>: %d tensors"), path, pf->ntensors);
}

const struct pth_tensor *pth_find(const struct pth_file *pf, const char *name)
{
    int i;

    for (i = 0; i < pf->ntensors; i++)
        if (strcmp(pf->tensors[i].name, name) == 0)
            return &pf->tensors[i];
    return NULL;
}

const struct pth_tensor *pth_require(const struct pth_file *pf,
                                     const char *name, int ndim,
                                     const long *shape)
{
    const struct pth_tensor *t = pth_find(pf, name);
    int d;

    if (!t)
        G_fatal_error(_("Tensor <%s> not found in checkpoint <%s>"), name,
                      pf->path);
    if (t->ndim != ndim)
        G_fatal_error(_("Tensor <%s> has rank %d, expected %d"), name, t->ndim,
                      ndim);
    for (d = 0; d < ndim; d++)
        if (shape && shape[d] >= 0 && t->shape[d] != shape[d])
            G_fatal_error(_("Tensor <%s> has size %ld along axis %d, "
                            "expected %ld"),
                          name, t->shape[d], d, shape[d]);
    return t;
}

long pth_numel(const struct pth_tensor *t)
{
    long n = 1;
    int d;

    for (d = 0; d < t->ndim; d++)
        n *= t->shape[d];
    return n;
}

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, u;
    float f;

    if (exp == 0) {
        if (man == 0)
            u = sign;
        else {
            /* Subnormal: renormalize into a float32 normal. */
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) {
                man <<= 1;
                exp--;
            }
            u = sign | (exp << 23) | ((man & 0x3ff) << 13);
        }
    }
    else if (exp == 31)
        u = sign | 0x7f800000 | (man << 13);
    else
        u = sign | ((exp + 127 - 15) << 23) | (man << 13);
    memcpy(&f, &u, 4);
    return f;
}

static float elem_to_float(const unsigned char *p, enum pth_dtype d)
{
    float f;
    uint32_t u;

    switch (d) {
    case PTH_F32:
        memcpy(&f, p, 4);
        return f;
    case PTH_F16:
        return half_to_float(rd16(p));
    default:
        u = (uint32_t)rd16(p) << 16;
        memcpy(&f, &u, 4);
        return f;
    }
}

float *pth_read_f32(const struct pth_file *pf, const struct pth_tensor *t)
{
    long n = pth_numel(t), expect = 1, i;
    size_t es = dtype_size(t->dtype);
    int d, contiguous = 1;
    float *out = G_malloc(n * sizeof(float));
    unsigned char *raw;

    for (d = t->ndim - 1; d >= 0; d--) {
        if (t->shape[d] != 1 && t->stride[d] != expect)
            contiguous = 0;
        expect *= t->shape[d];
    }

    if (contiguous) {
        raw = t->dtype == PTH_F32 ? (unsigned char *)out : G_malloc(n * es);
        read_at(pf->fp, t->data_offset + (off_t)t->storage_offset * es, raw,
                n * es, pf->path);
        if (t->dtype != PTH_F32) {
            for (i = 0; i < n; i++)
                out[i] = elem_to_float(raw + i * es, t->dtype);
            G_free(raw);
        }
        return out;
    }

    /* Strided view: read the whole storage and gather in C order. */
    raw = G_malloc(t->storage_bytes);
    read_at(pf->fp, t->data_offset, raw, t->storage_bytes, pf->path);
    {
        long idx[PTH_MAX_DIMS] = {0};

        for (i = 0; i < n; i++) {
            long off = t->storage_offset;

            for (d = 0; d < t->ndim; d++)
                off += idx[d] * t->stride[d];
            out[i] = elem_to_float(raw + off * es, t->dtype);
            for (d = t->ndim - 1; d >= 0; d--) {
                if (++idx[d] < t->shape[d])
                    break;
                idx[d] = 0;
            }
        }
    }
    G_free(raw);
    return out;
}

double pth_bytes(const struct pth_file *pf, const char *prefix)
{
    size_t len = strlen(prefix);
    double sum = 0;
    int i;

    for (i = 0; i < pf->ntensors; i++)
        if (strncmp(pf->tensors[i].name, prefix, len) == 0)
            sum += (double)pth_numel(&pf->tensors[i]) *
                   dtype_size(pf->tensors[i].dtype);
    return sum;
}

void pth_close(struct pth_file *pf)
{
    int i;

    if (pf->fp)
        fclose(pf->fp);
    for (i = 0; i < pf->ntensors; i++)
        G_free(pf->tensors[i].name);
    G_free(pf->tensors);
    G_free(pf->path);
    memset(pf, 0, sizeof(*pf));
}
