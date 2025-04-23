function val = get_field(params, fieldname, default_val)
    if isfield(params, fieldname)
        val = params.(fieldname);
    else
        val = default_val;
    end
end