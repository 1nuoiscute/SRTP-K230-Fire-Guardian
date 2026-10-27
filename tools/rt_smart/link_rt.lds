/* Linker script for RT-Smart big-core user programs, tuned for ld.lld.
 *
 * Derived from the SDK's userapps/sample/linker_scripts/riscv64/link.lds but
 * rewritten so ld.lld accepts it:
 *   - .plt is matched explicitly; the SDK's `*(*.plt)` glob also swallowed
 *     .rela.plt (SHT_RELA) into a SHT_PROGBITS output section, which lld rejects
 *   - __global_pointer$ is defined here (GNU ld supplies it implicitly, lld not)
 *   - sections keep the names and ordering symbols the SDK CRT/libraries expect
 *
 * The load address 0x200000000 is fixed: RT-Smart's LWP loader rejects entries
 * outside its user-space window (verified: entry 0x10000 gives
 * "[E/LWP] elf entry is invalidate").
 */

OUTPUT_ARCH( "riscv" )
ENTRY(_start)

SECTIONS
{
    . = 0x200000000;

    .text :
    {
        __text_start__ = .;
        *(.start)
        *(.text .text.*)
        KEEP(*(.init))
        KEEP(*(.fini))

        /* ctors / dtors, modern and legacy spellings */
        *(.ctors)
        *(SORT(.ctors.*))
        *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
        *(.dtors)
        *(SORT(.dtors.*))
        *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)

        *(.rodata .rodata.*)
        KEEP(*(.eh_frame*))

        . = ALIGN(8);
        _etext = .;
        __text_end__ = .;
    }

    . = ALIGN(0x1000);

    .plt :
    {
        *(.plt)
        *(.plt.got)
        *(.plt.sec)
        *(.iplt)
    }

    .got :
    {
        __got_start__ = .;
        *(.got .got.*)
        *(.got.plt)
        __got_end__ = .;
    }

    .data :
    {
        __data_start__ = .;

        *(vtable)
        *(.data .data.*)
        *(.sdata .sdata.*)
        *(.srodata .srodata.*)

        . = ALIGN(8);
        PROVIDE_HIDDEN(__preinit_array_start = .);
        KEEP(*(.preinit_array))
        PROVIDE_HIDDEN(__preinit_array_end = .);

        . = ALIGN(8);
        PROVIDE_HIDDEN(__init_array_start = .);
        KEEP(*(SORT(.init_array.*)))
        KEEP(*(.init_array))
        PROVIDE_HIDDEN(__init_array_end = .);

        . = ALIGN(8);
        PROVIDE_HIDDEN(__fini_array_start = .);
        KEEP(*(SORT(.fini_array.*)))
        KEEP(*(.fini_array))
        PROVIDE_HIDDEN(__fini_array_end = .);

        . = ALIGN(8);
        __data_end__ = .;
    }

    /* gp must sit in the middle of the small-data window so gp-relative
     * accesses reach both directions; ld.lld does not define it. */
    PROVIDE(__global_pointer$ = . + 0x800);

    .bss (NOLOAD) :
    {
        __bss_start__ = .;
        *(.sbss .sbss.*)
        *(.bss .bss.*)
        *(COMMON)
        *(.stack)
        . = ALIGN(8);
        __bss_end__ = .;
    }


    /* musl's crt1.o / __init_tls.c reference _DYNAMIC even in a fully static
     * link. An absolute 0 is unusable here: from a 0x200000000 base that is a
     * -2 GB displacement, outside R_RISCV_PCREL_HI20's range. A two-word zero
     * stub keeps the relocation local, and musl sees DT_NULL (0) as its first
     * d_tag, concluding there is no dynamic section. */
    .dynamic_stub :
    {
        __dynamic_stub = .;
        LONG(0);
        LONG(0);
    }
    PROVIDE(_DYNAMIC = __dynamic_stub);
    PROVIDE(_GLOBAL_OFFSET_TABLE_ = 0);
    . = ALIGN(16);
    _end = .;
    PROVIDE(end = .);
}

