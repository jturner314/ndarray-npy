/// Debug assertion that the value matches the pattern.
macro_rules! debug_assert_match {
    ($pattern:pat, $value:expr) => {
        if cfg!(debug_assertions) {
            #[allow(unreachable_patterns)]
            match $value {
                $pattern => {},
                _ => panic!("\
assertion failed: `(value matches pattern)`
 pattern: `{}`,
   value: `{:?}`", stringify!($pattern), $value),
            }
        }
    }
}

/// Extracts inner pairs matching the specified rules from the given pair.
///
/// Uses `debug_assert_match!` to assert that the inner pairs match the
/// specified rules. Uses `debug_assert_match!` to verify that there are no
/// additional pairs unless there is a `..` after the patterns (e.g.
/// `parse_pairs_as!(pair.into_inner(), (Rule::foo, Rule::bar | Rule::baz), ..)`).
///
/// For example:
///
/// ```rust,ignore
/// parse_pairs_as!(pair.into_inner(), (Rule::foo, Rule::bar))
/// ```
///
/// becomes
///
/// ```rust,ignore
/// {
///     let mut iter = &mut pair.into_inner();
///     let out = (
///         {
///             let tmp = iter.next().unwrap();
///             debug_assert_match!(Rule::foo, tmp.as_rule()),
///             tmp
///         },
///         {
///             let tmp = iter.next().unwrap();
///             debug_assert_match!(Rule::bar, tmp.as_rule()),
///             tmp
///         },
///     );
///     debug_assert_match!(None, iter.next());
///     out
/// }
/// ```
macro_rules! parse_pairs_as {
    // Entry point, fixed number of pairs, no trailing comma.
    ($pair:expr, ($($rules:pat),*)) => {
        parse_pairs_as!($pair, ($($rules),*,))
    };
    // Entry point, variable number of pairs, no trailing comma.
    ($pair:expr, ($($rules:pat),*), ..) => {
        parse_pairs_as!($pair, ($($rules),*,), ..)
    };
    // Entry point, fixed number of pairs, with trailing comma.
    ($pair:expr, ($($rules:pat),*,)) => {
        {
            let iter = &mut $pair;
            let out = parse_pairs_as!(@recur iter, (), ($($rules),*,));
            debug_assert_match!(Option::None, iter.next());
            out
        }
    };
    // Entry point, variable number of pairs, with trailing comma.
    ($pair:expr, ($($rules:pat),*,), ..) => {
        {
            let iter = &mut $pair;
            parse_pairs_as!(@recur iter, (), ($($rules),*,))
        }
    };
    // Start processing the patterns.
    (@recur
        $iter:ident,
        (),
        ($head_rule:pat, $($remaining_rules:pat),*,)
    ) => {
        parse_pairs_as!(@recur
            $iter,
            (
                {
                    let tmp = $iter.next().unwrap();
                    debug_assert_match!($head_rule, tmp.as_rule());
                    tmp
                },
            ),
            ($($remaining_rules),*,)
        )
    };
    // Continue processing the patterns.
    (@recur
        $iter:ident,
        ($($processed:expr),*,),
        ($head_rule:pat, $($remaining_rules:pat),*,)
    ) => {
        parse_pairs_as!(@recur
            $iter,
            (
                $($processed),*,
                {
                    let tmp = $iter.next().unwrap();
                    debug_assert_match!($head_rule, tmp.as_rule());
                    tmp
                },
            ),
            ($($remaining_rules),*,)
        )
    };
    // Start processing the (single) pattern.
    (@recur $iter:ident, (), ($head_rule:pat,)) => {
        (
            {
                let tmp = $iter.next().unwrap();
                debug_assert_match!($head_rule, tmp.as_rule());
                tmp
            },
        )
    };
    // Finish processing the patterns.
    (@recur $iter:ident, ($($processed:expr),*,), ($head_rule:pat,)) => {
        (
            $($processed),*,
            {
                let tmp = $iter.next().unwrap();
                debug_assert_match!($head_rule, tmp.as_rule());
                tmp
            },
        )
    };
}
